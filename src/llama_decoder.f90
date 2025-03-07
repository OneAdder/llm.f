! Qwen Python reference: https://github.com/OneAdder/neural-fortran-references/blob/main/qwen_decoder.py
! Llama Pytohn reference: https://github.com/OneAdder/neural-fortran-references/blob/main/llama_decoder.py
module llmf_llama_decoder
  use iso_fortran_env, only: stderr => error_unit
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer
  use llmf_rmsnorm, only: rmsnorm_layer
  use llmf_llama_attention, only: llama_attention_layer
  use llmf_llama_feed_forward, only: llama_feed_forward_layer
  implicit none

  private
  public :: llama_decoder_layer

  type, extends(base_layer) :: llama_decoder_layer
    logical :: is_qwen
    integer :: sequence_length, model_dimension, intermediate_size, n_heads, n_kv_heads

    type(rmsnorm_layer) :: input_layernorm
    type(llama_attention_layer) :: self_attn
    type(llama_feed_forward_layer) :: feed_forward
    type(rmsnorm_layer) :: post_attention_layernorm

    real, allocatable :: output(:, :)
    real, allocatable :: gradient(:, :)

    ! it is the second residual, the first residual doesn't need to be copied
    real, allocatable, private :: residual(:, :)
    real, allocatable, private :: d_residual(:, :)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type llama_decoder_layer

  interface llama_decoder_layer
    module function llama_decoder_layer_cons(n_heads, n_kv_heads, intermediate_size, is_qwen) result(res)
      integer :: n_heads, n_kv_heads, intermediate_size
      logical, optional, intent(in) :: is_qwen
      type(llama_decoder_layer) :: res
    end function llama_decoder_layer_cons
  end interface llama_decoder_layer

contains
  module function llama_decoder_layer_cons(n_heads, n_kv_heads, intermediate_size, is_qwen) result(res)
    integer :: n_heads, n_kv_heads, intermediate_size
    logical, optional, intent(in) :: is_qwen
    type(llama_decoder_layer) :: res

    res % n_heads = n_heads
    res % n_kv_heads = n_kv_heads
    res % intermediate_size = intermediate_size
    if (present(is_qwen)) then
      res % is_qwen = is_qwen
    else
      res % is_qwen = .false.
    end if
  end function llama_decoder_layer_cons

  module subroutine forward(self, input, cosine, sine, attention_mask)
    class(llama_decoder_layer), intent(inout) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: cosine(:, :)
    real, intent(in) :: sine(:, :)
    real, intent(in) :: attention_mask(:, :)

    integer :: i, j

    call self % input_layernorm % forward(input)
    call self % self_attn % forward(self % input_layernorm % output, cosine, sine, attention_mask)
    self % residual = self % self_attn % output + input

    call self % post_attention_layernorm % forward(self % residual)
    call self % feed_forward % forward(self % post_attention_layernorm % output)

    self % output = self % feed_forward % output + self % residual
  end subroutine forward

  module subroutine backward(self, input, gradient, cosine, sine, attention_mask)
    class(llama_decoder_layer), intent(inout) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    real, intent(in) :: cosine(:, :)
    real, intent(in) :: sine(:, :)
    real, intent(in) :: attention_mask(:, :)

    call self % feed_forward % backward(self % post_attention_layernorm % output, gradient)
    call self % post_attention_layernorm % backward(self % residual, self % feed_forward % gradient)

    self % d_residual = self % post_attention_layernorm % gradient + gradient
    call self % self_attn % backward(&
        self % input_layernorm % output,&
        self % d_residual,&
        cosine, sine, attention_mask&
    )
    call self % input_layernorm % backward(input, self % self_attn % gradient)

    self % gradient = self % input_layernorm % gradient + self % d_residual
  end subroutine backward

  module subroutine init(self, input_shape)
    class(llama_decoder_layer), intent(inout) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "Llama Decoder accepts 2D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    self % input_layernorm = rmsnorm_layer()
    call self % input_layernorm % init([self % sequence_length, self % model_dimension])

    self % self_attn = llama_attention_layer(self % n_heads, self % n_kv_heads, self % is_qwen)
    call self % self_attn % init([self % sequence_length, self % model_dimension])

    self % feed_forward = llama_feed_forward_layer(self % intermediate_size)
    call self % feed_forward % init([self % sequence_length, self % model_dimension])

    self % post_attention_layernorm = rmsnorm_layer()
    call self % post_attention_layernorm % init([self % sequence_length, self % model_dimension])

    allocate(self % residual, mold=self % output)
    allocate(self % d_residual, mold=self % output)
  end subroutine init
end module llmf_llama_decoder
