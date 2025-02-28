module llmf_rmsnorm
  use nf_base_layer, only: base_layer
  implicit none

  type, extends(base_layer) :: rmsnorm_layer
    integer :: sequence_length, model_dimension
    real :: eps
    real, allocatable :: gamma(:)
    real, allocatable :: output(:, :)
  contains
    procedure :: forward
    procedure :: init
  end type rmsnorm_layer

  interface rmsnorm_layer
    module function rmsnorm_layer_cons() result(res)
      type(rmsnorm_layer) :: res
    end function rmsnorm_layer_cons
  end interface rmsnorm_layer

contains
  module function rmsnorm_layer_cons() result(res)
    type(rmsnorm_layer) :: res

    res % eps = 1e-5
  end function rmsnorm_layer_cons

  subroutine forward(self, input)
    class(rmsnorm_layer), intent(inout) :: self
    real :: input(:, :)
    integer :: i

    do concurrent(i = 1: self % sequence_length)
      self % output(i, :) = &
          (input(i, :) * self % gamma) &
          / sqrt(self % eps + (sum(input(i, :) ** 2) / size(input, 2)))
    end do
  end subroutine forward

  module subroutine init(self, input_shape)
    class(rmsnorm_layer), intent(inout) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "RMSNorm Layer accepts 2D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    allocate(self % output(self % sequence_length, self % model_dimension))
    ! default initialization from PyTorch
    allocate(self % gamma(self % model_dimension))
    self % gamma = 1.
  end subroutine init
end module llmf_rmsnorm