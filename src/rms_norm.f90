module llmf_rmsnorm
  use nf_base_layer, only: base_layer
  implicit none

  type, extends(base_layer) :: rmsnorm_layer
    integer :: sequence_length, model_dimension
    real :: eps
    real, allocatable :: gamma(:)
    real, allocatable :: gradient(:, :)
    real, allocatable :: dw(:)
    real, allocatable :: output(:, :)

    real, allocatable, private :: one_over_sigma(:, :)
    real, allocatable, private :: gradient_by_gamma_over_sigma(:)
    real, allocatable, private :: weight_deltas(:, :)
  contains
    procedure :: forward
    procedure :: backward
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
      self % one_over_sigma(i, :) = 1 / sqrt(self % eps + (sum(input(i, :) ** 2) / size(input, 2)))
      self % output(i, :) = input(i, :) * self % gamma * self % one_over_sigma(i, :)
    end do
  end subroutine forward

  subroutine backward(self, input, gradient)
    class(rmsnorm_layer), intent(inout) :: self
    real :: input(:, :)
    real :: gradient(:, :)
    integer :: i

    do concurrent(i = 1: self % sequence_length)
      self % gradient_by_gamma_over_sigma = gradient(i, :) * self % gamma * self % one_over_sigma(i, :)

      self % weight_deltas(i, :) = gradient(i, :) * input(i, :) * self % one_over_sigma(i, :)
      self % gradient(i, :) = self % gradient_by_gamma_over_sigma &
        - (&
              sum(input(i, :) * self % gradient_by_gamma_over_sigma * (self % one_over_sigma(i, :) ** 2))&
          ) * (input(i, :) / self % model_dimension)
    end do

    self % dw = sum(self % weight_deltas, dim=1)
  end subroutine backward

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

    allocate(self % gradient(self % sequence_length, self % model_dimension))
    allocate(self % dw(self % model_dimension))

    ! allocate temp storages
    allocate(self % one_over_sigma(self % sequence_length, self % model_dimension))
    allocate(self % gradient_by_gamma_over_sigma(self % model_dimension))
    allocate(self % weight_deltas, mold=self % gradient)
  end subroutine init
end module llmf_rmsnorm
