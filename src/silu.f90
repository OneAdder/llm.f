#define _sigmoid(x) 1 / (1 + exp(-x))

module llmf_silu
  use nf_base_layer, only: base_layer
  implicit none

  type, extends(base_layer) :: silu_layer
    integer :: model_dimension
    real, allocatable :: gradient(:)
    real, allocatable :: output(:)

    real, allocatable, private :: sigmoid_x(:)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type silu_layer

  interface silu_layer
    module function silu_layer_cons() result(res)
      type(silu_layer) :: res
    end function silu_layer_cons
  end interface silu_layer

contains
  module function silu_layer_cons() result(res)
    type(silu_layer) :: res
  end function silu_layer_cons

  subroutine forward(self, input)
    class(silu_layer), intent(inout) :: self
    real :: input(:)

    self % sigmoid_x = _sigmoid(input)
    self % output = input * self % sigmoid_x
  end subroutine forward

  subroutine backward(self, input, gradient)
    class(silu_layer), intent(inout) :: self
    real :: input(:)
    real :: gradient(:)

    self % gradient = gradient * (input * self % sigmoid_x * (1. - self % sigmoid_x) + self % sigmoid_x)
  end subroutine backward

  module subroutine init(self, input_shape)
    class(silu_layer), intent(inout) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 1) then
      error stop "Silu Layer accepts 1D input"
    end if
    self % model_dimension = input_shape(1)

    allocate(self % output(self % model_dimension))
    allocate(self % gradient(self % model_dimension))

    ! allocate temp storage for sigmoid output caching
    allocate(self % sigmoid_x, mold=self % output)
  end subroutine init
end module llmf_silu
