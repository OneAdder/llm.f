module llmf_llama_attention
  use iso_fortran_env, only: stderr => error_unit
  use nf_multihead_attention_layer, only: multihead_attention_layer
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  private
  public :: llama_attention_layer

  type, extends(multihead_attention_layer) :: llama_attention_layer
    integer :: n_kv_heads, n_kv_groups
    real, allocatable :: q_temp(:, :, :)
    real, allocatable :: k_temp(:, :, :)
    real, allocatable :: v_temp(:, :, :)
  contains
    procedure :: forward
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
    self % k_or_dk = reshape(&
        spread(self % k_temp, dim=3, ncopies=self % n_kv_groups),&
        shape(self % q_temp)&
    )
    ! repeat groups for key to total amount of heads
    self % v_or_dv = reshape(&
        spread(self % v_temp, dim=3, ncopies=self % n_kv_groups),&
        shape(self % q_temp)&
    )

    call self % sdpa_forward(attention_mask)
  end subroutine forward

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

    allocate(self % q_temp(self % sequence_length, self % head_size, self % n_heads))
    allocate(self % k_temp(self % sequence_length, self % head_size, self % n_kv_heads))
    allocate(self % v_temp(self % sequence_length, self % head_size, self % n_kv_heads))
  end subroutine init
end module llmf_llama_attention
