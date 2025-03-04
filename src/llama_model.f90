!! Python Reference: https://github.com/OneAdder/neural-fortran-references/blob/main/qwen_model.py
module llmf_llama_model
  use nf_base_layer, only: base_layer
  use nf_embedding_layer, only: embedding_layer
  use llmf_rope, only: rotary_embedding
  use llmf_llama_decoder, only: llama_decoder_layer
  use llmf_rmsnorm, only: rmsnorm_layer
  implicit none

  type, extends(base_layer) :: llama_model
    integer :: batch_size, sequence_length, intermediate_size, model_dimension
    integer :: vocab_size, n_layers, n_heads, n_kv_heads
    logical :: is_qwen

    real, allocatable :: gradient(:, :, :)
    real, pointer :: output(:, :, :)

    type(embedding_layer) :: embed_tokens
    type(rotary_embedding) :: rope
    type(llama_decoder_layer), allocatable :: decoder_stack(:)
    type(rmsnorm_layer) :: norm

  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type llama_model

  interface llama_model
    module function llama_model_cons(&
        vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size, is_qwen&
    ) result(res)
      integer, intent(in) :: vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size
      logical, optional :: is_qwen
      type(llama_model) :: res
    end function llama_model_cons
  end interface llama_model

contains
  module function llama_model_cons(&
      vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size, is_qwen&
  ) result(res)
    integer, intent(in) :: vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size
    logical, optional :: is_qwen
    type(llama_model) :: res

    res % vocab_size = vocab_size
    res % n_layers = n_layers
    res % n_heads = n_heads
    res % n_kv_heads = n_kv_heads
    res % intermediate_size = intermediate_size
    if (present(is_qwen)) then
      res % is_qwen = is_qwen
    else
      res % is_qwen = .false.
    end if
  end function llama_model_cons

  subroutine forward(self, input, attention_mask)
    class(llama_model), intent(inout), target :: self
    integer :: input(:, :)
    logical :: attention_mask(:, :)
    integer :: batch, layer
    real :: embedding_output(self % sequence_length, self % model_dimension, self % batch_size)
    real :: cosine(self % sequence_length, self % rope % head_size)
    real :: sine(self % sequence_length, self % rope % head_size)
    real :: decoder_stack_output(self % sequence_length, self % model_dimension, self % batch_size)

    do batch = 1, self % batch_size
      call self % embed_tokens % forward(input(:, batch))
      embedding_output(:, :, batch) = self % embed_tokens % output

      call self % rope % apply([0, 1, 2, 3, 4, 5, 6, 7, 8], cosine, sine)
      call self % decoder_stack(1) % forward(embedding_output(:, :, batch), cosine, sine)
      decoder_stack_output(:, :, batch) = self % decoder_stack(1) % output
      do layer = 2, self % n_layers
        call self % decoder_stack(layer) % forward(decoder_stack_output(:, :, batch), cosine, sine)
        decoder_stack_output(:, :, batch) = self % decoder_stack(layer) % output
      end do
      call self % norm % forward(self % decoder_stack(self % n_layers) % output)
      self % output(:, :, batch) = self % norm % output
    end do
  end subroutine forward

  subroutine backward(self, input, gradient, attention_mask)
    class(llama_model), intent(inout), target :: self
    integer :: input(:, :)
    real :: gradient(:, :)
    logical :: attention_mask(:, :)

  end subroutine backward

  module subroutine init(self, input_shape)
    class(llama_model), intent(inout) :: self
    integer, intent(in) :: input_shape(:)
    integer :: i

    if (size(input_shape) /= 3) then
      error stop "Llama Model accepts 3D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)
    self % batch_size = input_shape(3)

    ! init layers
    self % embed_tokens = embedding_layer(self % vocab_size, self % model_dimension)
    call self % embed_tokens % init([self % sequence_length])

    self % rope = rotary_embedding(&
        self % sequence_length, self % model_dimension, self % model_dimension / self % n_heads&
    )

    allocate(self % decoder_stack(self % n_layers))
    do i = 1, self % n_layers
      self % decoder_stack(i) = llama_decoder_layer(&
          intermediate_size=self % intermediate_size,&
          n_heads=4,&
          n_kv_heads=2,&
          is_qwen=self % is_qwen&
      )
      call self % decoder_stack(i) % init([self % sequence_length, self % model_dimension])
    end do

    self % norm = rmsnorm_layer()
    call self % norm % init([self % sequence_length, self % model_dimension])

    ! allocate public
    allocate(self % output(self % sequence_length, self % model_dimension, self % batch_size))
    allocate(self % gradient(self % sequence_length, self % model_dimension, self % batch_size))
  end subroutine init
end module llmf_llama_model
