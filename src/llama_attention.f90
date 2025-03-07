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
    logical :: is_qwen

    real, allocatable :: gradient(:, :)
    real, allocatable :: q_temp(:, :, :)
    real, allocatable :: k_temp(:, :, :)
    real, allocatable :: v_temp(:, :, :)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: repeat_interleave
    procedure :: repeat_interleave_backward
    procedure :: apply_rotary_pos_emb
    procedure :: apply_rotary_pos_emb_backward
    procedure :: rotate_half
    procedure :: combine_kv_heads
    procedure :: init
  end type llama_attention_layer

  interface llama_attention_layer
    module function llama_attention_layer_cons(n_heads, n_kv_heads, is_qwen) result(res)
      integer, intent(in) :: n_heads, n_kv_heads
      logical, optional, intent(in) :: is_qwen
      type(llama_attention_layer) :: res
    end function llama_attention_layer_cons
  end interface llama_attention_layer

contains
  module function llama_attention_layer_cons(n_heads, n_kv_heads, is_qwen) result(res)
    integer, intent(in) :: n_heads, n_kv_heads
    logical, optional, intent(in) :: is_qwen
    type(llama_attention_layer) :: res
    res % n_heads = n_heads
    res % n_kv_heads = n_kv_heads
    if (present(is_qwen)) then
      res % is_qwen = is_qwen
    else
      res % is_qwen = .false.
    end if
  end function llama_attention_layer_cons

  module subroutine forward(self, input, cosine, sine, attention_mask)
    !! Args:
    !! input: input of Self Attention (sequence_length, model_dimension)
    !! cosine: cos values of positional encoding (sequence_length, head_size)
    !! sine: sine values of positional encoding (sequence_length, head_size)
    !! attention_mask: attention mask, floating point, not bool (sequence_length, sequence_length)
    !! What it does:
    !! 1. Copy inputs into temp storages (similar to `torch.ctx`)
    !! 2. Forward through in projections
    !! 3. Split attention heads for query and key-value heads for key and value
    !! 4. Apply ropes for query and key
    !! 5. Repeat key-value to the amount to attention heads for key and value
    !! 6. Store results in `q_or_dq`, `k_or_dk` and `v_or_dv` temp variables
    !! 7. Forward through Scaled Dot Product Attention
    !! Graph:
    !!        input
    !!     /    |     \
    !! query   key    value
    !!   |      |       |
    !!    linear_forward
    !!   |      |       |
    !! split     split_kv --store--> q_heads k_heads v_heads
    !!   |      |       |
    !!  apply_ropes <---|---cos,sin
    !!   |      |       |
    !!   |   repeat_kv_heads
    !!   |      |       |
    !! scaled_dot_product_attention < attention_mask
    !!   \      |      /
    !!        output
    class(llama_attention_layer), intent(inout) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: cosine(:, :)
    real, intent(in) :: sine(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    real :: k(self % sequence_length, self % head_size, self % n_kv_heads)
    real :: v(self % sequence_length, self % head_size, self % n_kv_heads)

    self % q_input = input
    self % k_input = input
    self % v_input = input

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

    self % q_or_dq = self % q_temp
    ! repeat groups for key to total amount of heads
    self % k_or_dk = self % repeat_interleave(self % k_temp)
    ! repeat groups for key to total amount of heads
    self % v_or_dv = self % repeat_interleave(self % v_temp)

    call self % sdpa_forward(attention_mask)
  end subroutine forward

  module subroutine backward(self, input, gradient, cosine, sine, attention_mask)
    !! What it does:
    !! 1. Repeat key and value input (before ropes) to the amount of attention heads
    !! 2. Backward through Scaled Dot Product Attention
    !! 3. Repeat backward (partial sum) for key
    !! 4. Backward through ropes (cos - sin)
    !! 5. Repeat backward (partial sum) for value
    !! 6. Combine heads
    !! 7. Backward through in projections
    !! 8. Sum gradients for query, key and value as it is Self Attention
    !! Graph:
    !!    |
    !! gradient  q_heads k_heads v_heads <--from-stored--
    !!    |       |        |       |
    !!    |       |      repeat_kv_heads
    !!    |       |        |       |
    !! scaled_dot_product_attention_backward < attention_mask
    !!    |       |       |
    !!   dq      dk      dv
    !!    |       |       |
    !!    |    repeat_backward
    !!    |       |       |
    !!  ropes_backward <--|---cos,sin
    !!    |       |       |
    !!     linear_backward
    !!     \      |      /
    !!       output = sum
    class(llama_attention_layer), intent(inout) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    real, intent(in) :: cosine(:, :)
    real, intent(in) :: sine(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    self % v_heads = self % repeat_interleave(self % v_temp)
    self % k_heads = self % repeat_interleave(self % k_temp)
    self % q_heads = self % q_temp

    call self % sdpa_backward(gradient, attention_mask)

    self % k_temp = self % repeat_interleave_backward(self % k_or_dk)
    self % q_temp = self % q_or_dq
    call self % apply_rotary_pos_emb_backward(self % q_temp, self % k_temp, cosine, sine)

    call self % value_layer % backward(&
        self % v_input,&
        self % combine_kv_heads(self % repeat_interleave_backward(self % v_or_dv))&
    )
    call self % key_layer % backward(&
        self % k_input,&
        self % combine_kv_heads(self % k_temp)&
    )
    call self % query_layer % backward(&
        self % q_input,&
        self % combine_heads(self % q_temp)&
    )

    self % gradient = &
        self % query_layer % gradient &
        + self % key_layer % gradient &
        + self % value_layer % gradient
  end subroutine backward

  module function repeat_interleave(self, input) result(res)
    ! repeat by dimension and write result into that dimension
    class(llama_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :, :)
    real :: res(size(input, 1), size(input, 2), size(input, 3) * self % n_kv_groups)

    res = reshape(&
        spread(input, dim=3, ncopies=self % n_kv_groups),&
        shape(res)&
    )
  end function repeat_interleave

  module function repeat_interleave_backward(self, gradient) result(res)
    ! backwards pass for `repeat_interleave`
    ! Example for `gradient` shape (3, 4, 6):
    ! 1. Split last dimension into three parts (last_graient_dim / self % n_kv_groups = 6 / 2 = 3),
    !    intermediate shape = (3, 2)
    ! 2. Sum over the first dim of intermediate shape, getting array of size (2)
    ! 3. Use this array as the last dimension of output, resulting in (3, 4, 2)
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

  module subroutine apply_rotary_pos_emb_backward(self, query, key, cosine, sine)
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
      query(:, :, head) = (query(:, :, head) * cosine) - (q_rotated(:, :, head) * sine)
    end do

    k_rotated = self % rotate_half(key, self % n_kv_heads)
    do concurrent(head = 1: self % n_kv_heads)
      key(:, :, head) = (key(:, :, head) * cosine) - (k_rotated(:, :, head) * sine)
    end do
  end subroutine apply_rotary_pos_emb_backward

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

  module function combine_kv_heads(self, input) result(output)
    ! usual `combine_heads` will not work as amount of kv_heads and attention heads may be different
    class(llama_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :, :)
    real :: output(self % sequence_length, self % n_kv_groups * self % n_kv_heads)
    integer :: seq

    do concurrent(seq = 1: self % sequence_length)
      output(seq, :) = reshape(transpose(input(seq, :, :)), [size(output, 2)])
    end do
  end function combine_kv_heads

  module subroutine init(self, input_shape)
    class(llama_attention_layer), intent(inout) :: self
    integer, intent(in) :: input_shape(:)

    call self % init_base(input_shape)

    if (mod(self % n_heads, self % n_kv_heads) /= 0) then
      write(stderr, '(a)') 'number of key-value heads must be divisible by number of attention heads'
      error stop 1
    end if
    self % n_kv_groups = self % n_heads / self % n_kv_heads

    ! projection layers differ from usual MHA, so they need to be reinitialized
    if (self % is_qwen) then
      ! Qwen has biases for QKV
      self % query_layer = linear2d_layer(self % model_dimension, biases=.true.)
      self % key_layer = linear2d_layer(self % n_kv_heads * self % head_size, biases=.true.)
      self % value_layer = linear2d_layer(self % n_kv_heads * self % head_size, biases=.true.)
    else
      ! Llama doesn't
      self % query_layer = linear2d_layer(self % model_dimension, biases=.false.)
      self % key_layer = linear2d_layer(self % n_kv_heads * self % head_size, biases=.false.)
      self % value_layer = linear2d_layer(self % n_kv_heads * self % head_size, biases=.false.)
    end if
    ! neither use biases for output
    self % output_layer = linear2d_layer(self % model_dimension, biases=.false.)

    call self % query_layer % init([self % sequence_length, self % model_dimension])
    call self % key_layer % init([self % sequence_length, self % model_dimension])
    call self % value_layer % init([self % sequence_length, self % model_dimension])
    call self % output_layer % init([self % sequence_length, self % model_dimension])

    allocate(self % gradient(self % sequence_length, self % model_dimension))

    allocate(self % q_temp(self % sequence_length, self % head_size, self % n_heads))
    allocate(self % k_temp(self % sequence_length, self % head_size, self % n_kv_heads))
    allocate(self % v_temp(self % sequence_length, self % head_size, self % n_kv_heads))
  end subroutine init
end module llmf_llama_attention
