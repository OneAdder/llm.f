module llmf_rope
  implicit none

  type :: rotary_embedding
    integer :: sequence_length, model_dimension, max_position_embeddings, head_size
    real :: rope_theta, rope_scaling
    real, allocatable :: inv_freq(:)
    real, allocatable :: freqs(:, :)
  contains
    procedure :: apply
  end type rotary_embedding

  interface rotary_embedding
    module function rotary_embedding_cons(sequence_length, model_dimension, head_size) result(res)
      integer, intent(in) :: sequence_length, model_dimension, head_size
      type(rotary_embedding) :: res
    end function rotary_embedding_cons
  end interface rotary_embedding

contains
  module function rotary_embedding_cons(sequence_length, model_dimension, head_size) result(res)
    integer, intent(in) :: sequence_length, model_dimension, head_size
    type(rotary_embedding) :: res
    integer :: i, dim

    res % sequence_length = sequence_length
    res % model_dimension = model_dimension
    res % head_size = head_size

    res % rope_theta = 10000.
    res % rope_scaling = 1.

    allocate(res % inv_freq(sequence_length))
    res % inv_freq = 1. / (res % rope_theta ** ([(i, i=0, head_size-1, 2)] / real(head_size)))

    allocate(res % freqs(sequence_length, head_size / 2))
  end function rotary_embedding_cons

  pure module subroutine apply(self, position_ids, cosine, sine)
    class(rotary_embedding), intent(inout) :: self
    integer, intent(in) :: position_ids(:)
    real, intent(inout) :: cosine(:, :)
    real, intent(inout) :: sine(:, :)

    self % freqs = transpose(matmul(&
        reshape(self % inv_freq, [self % head_size / 2, 1]),&
        reshape(position_ids, [1, self % sequence_length])&
    ))

    cosine = reshape(&
        spread(cos(self % freqs), dim=3, ncopies=2),&
        [self % sequence_length, self % head_size]&
    )
    sine = reshape(&
        spread(sin(self % freqs), dim=3, ncopies=2),&
        [self % sequence_length, self % head_size]&
    )
  end subroutine apply
end module llmf_rope
