module llmf_llama_attention
  use iso_fortran_env, only: stderr => error_unit
  use nf_multihead_attention_layer, only: multihead_attention_layer
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  private
  public :: llama_attention_layer

  type, extends(multihead_attention_layer) :: llama_attention_layer
    !! Python Reference: https://github.com/OneAdder/neural-fortran-references/blob/main/llama_attention.py
    integer :: n_kv_heads, n_kv_groups
    real, allocatable :: gradient(:, :)
    real, allocatable :: q_temp(:, :, :)
    real, allocatable :: k_temp(:, :, :)
    real, allocatable :: v_temp(:, :, :)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: repeat_interleave
    procedure :: repeat_interleave_backward
    procedure :: combine_kv_heads
    procedure :: apply_rotary_pos_emb
    procedure :: rotate_half
    procedure :: init => init
  end type llama_attention_layer

  interface llama_attention_layer
    module function llama_attention_layer_cons(n_heads, n_kv_heads) result(res)
      integer, intent(in) :: n_heads, n_kv_heads
      type(llama_attention_layer) :: res
    end function llama_attention_layer_cons
  end interface llama_attention_layer

contains
  module function llama_attention_layer_cons(n_heads, n_kv_heads) result(res)
    integer, intent(in) :: n_heads, n_kv_heads
    type(llama_attention_layer) :: res
    res % n_heads = n_heads
    res % n_kv_heads = n_kv_heads
  end function llama_attention_layer_cons

  module subroutine backward(self, input, gradient, cosine, sine, attention_mask)
    class(llama_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    real, intent(in) :: cosine(:, :)
    real, intent(in) :: sine(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    self % v_heads = self % v_temp
    self % k_heads = self % k_temp
    self % q_heads = self % q_temp

    call self % sdpa_backward(gradient, attention_mask)

    self % k_temp = self % repeat_interleave_backward(self % k_or_dk)

    ! FIXME: implement backward for apply_rotary_pos_emb
    call self % apply_rotary_pos_emb(self % q_or_dq, self % k_temp, cosine, sine)

    call self % value_layer % backward(&
        self % v_input,&
        self % combine_kv_heads(self % repeat_interleave_backward(self % v_or_dv))&
    )
    call self % key_layer % backward(&
        self % k_input,&
        self % combine_kv_heads(self % k_temp)&
    )
    call self % query_layer % backward(self % q_input, self % combine_heads(self % q_or_dq))

    self % gradient = &
        self % query_layer % gradient &
        + self % key_layer % gradient &
        + self % value_layer % gradient
  end subroutine backward

  module function repeat_interleave_backward(self, gradient) result(res)
    class(llama_attention_layer), intent(in) :: self
    real, intent(in) :: gradient(:, :, :)
    real :: res(size(gradient, 1), size(gradient, 2), size(gradient, 3) / self % n_kv_groups)
    integer :: reshape_size(2)
    integer :: i, j

    reshape_size = [size(gradient, 3) / self % n_kv_groups, self % n_kv_groups]

    do concurrent(i = 1: self % sequence_length, j = 1: self % head_size)
      res(i, j, :) = sum(&
          reshape(gradient(i, j, :), reshape_size),&
          dim=1&
      )
    end do
  end function repeat_interleave_backward

  module function combine_kv_heads(self, input) result(output)
    class(llama_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :, :)
    real :: output(self % sequence_length, self % n_kv_groups * self % n_kv_heads)
    integer :: seq

    do concurrent(seq = 1: self % sequence_length)
      output(seq, :) = reshape(transpose(input(seq, :, :)), [size(output, 2)])
    end do
  end function combine_kv_heads

  module subroutine forward(self, input, cosine, sine, attention_mask)
    !! input: input of Self Attention (sequence_length, model_dimension)
    !! cosine: cos values of positional encoding (sequence_length, head_size)
    !! sine: sine values of positional encoding (sequence_length, head_size)
    !! attention_mask: attention mask, floating point, not bool (sequence_length, sequence_length)
    class(llama_attention_layer), intent(inout) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: cosine(:, :)
    real, intent(in) :: sine(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    real :: k(self % sequence_length, self % head_size, self % n_kv_heads)
    real :: v(self % sequence_length, self % head_size, self % n_kv_heads)

    call self % query_layer % forward(input)
    call self % key_layer % forward(input)
    call self % value_layer % forward(input)

    self % q_temp = self % split_heads(self % query_layer % output)
    self % k_temp = reshape(&
        self % key_layer % output,&
        [self % sequence_length, self % head_size, self % n_kv_heads]&
    )
    self % v_temp = reshape(&
        self % value_layer % output,&
        [self % sequence_length, self % head_size, self % n_kv_heads]&
    )

    call self % apply_rotary_pos_emb(query=self % q_temp, key=self % k_temp, cosine=cosine, sine=sine)

    ! FIXME: added for memory safety, remove when backward is tested and works without it
    self % q_or_dq = self % q_temp
    ! repeat groups for key to total amount of heads
    self % k_or_dk = self % repeat_interleave(self % k_temp)
    ! repeat groups for key to total amount of heads
    self % v_or_dv = self % repeat_interleave(self % v_temp)

    call self % sdpa_forward(attention_mask)
  end subroutine forward

  module function repeat_interleave(self, input) result(res)
    class(llama_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :, :)
    real :: res(size(input, 1), size(input, 2), size(input, 3) * self % n_kv_groups)

    res = reshape(&
        spread(input, dim=3, ncopies=self % n_kv_groups),&
        shape(res)&
    )
  end function repeat_interleave

  module subroutine apply_rotary_pos_emb(self, query, key, cosine, sine)
    class(llama_attention_layer), intent(inout) :: self
    real, intent(inout) :: query(:, :, :)
    !! (sequence_length, head_size, n_heads)
    real, intent(inout) :: key(:, :, :)
    !! (sequence_length, head_size, n_kv_heads)
    real, intent(in) :: cosine(:, :)
    !! (sequence_length, head_size)
    real, intent(in) :: sine(:, :)
    !! (sequence_length, head_size)

    real :: q_rotated(self % sequence_length, self % head_size, self % n_heads)
    real :: k_rotated(self % sequence_length, self % head_size, self % n_kv_heads)
    integer :: head, head_dim

    q_rotated = self % rotate_half(query, self % n_heads)
    do concurrent(head = 1: self % n_heads)
      query(:, :, head) = (query(:, :, head) * cosine) + (q_rotated(:, :, head) * sine)
    end do

    k_rotated = self % rotate_half(key, self % n_kv_heads)
    do concurrent(head = 1: self % n_kv_heads)
      key(:, :, head) = (key(:, :, head) * cosine) + (k_rotated(:, :, head) * sine)
    end do
  end subroutine apply_rotary_pos_emb

  pure module function rotate_half(self, input, dim) result(res)
    !! Split head size by two, negate the second part, swap the two parts
    class(llama_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :, :)
    integer, intent(in) :: dim
    real :: res(self % sequence_length, self % head_size, dim)
    integer :: half, seq, head

    half = size(input, 2) / 2

    res(:, 1: half, :) = -input(:, half+1: self % head_size, :)
    res(:, half+1: self % head_size, :) = input(:, 1: half, :)
  end function rotate_half

  module subroutine init(self, input_shape)
    class(llama_attention_layer), intent(inout) :: self
    integer, intent(in) :: input_shape(:)

    call self % init_base(input_shape)

    if (mod(self % n_heads, self % n_kv_heads) /= 0) then
      write(stderr, '(a)') 'number of key-value heads must be divisible by number of attention heads'
      error stop 1
    end if
    self % n_kv_groups = self % n_heads / self % n_kv_heads

    !! key and value layers differ from usual MHA, so they need to be reinitialized
    self % key_layer = linear2d_layer(self % n_kv_heads * self % head_size)
    call self % key_layer % init([self % sequence_length, self % model_dimension])
    self % value_layer = linear2d_layer(self % n_kv_heads * self % head_size)
    call self % value_layer % init([self % sequence_length, self % model_dimension])

    allocate(self % gradient(self % sequence_length, self % model_dimension))

    allocate(self % q_temp(self % sequence_length, self % head_size, self % n_heads))
    allocate(self % k_temp(self % sequence_length, self % head_size, self % n_kv_heads))
    allocate(self % v_temp(self % sequence_length, self % head_size, self % n_kv_heads))

    deallocate(self % k_heads)
    allocate(self % k_heads(self % sequence_length, self % head_size, self % n_kv_heads))
    deallocate(self % v_heads)
    allocate(self % v_heads(self % sequence_length, self % head_size, self % n_kv_heads))
  end subroutine init
end module llmf_llama_attention
