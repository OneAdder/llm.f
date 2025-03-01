module llmf_swiglu
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer
  use llmf_silu, only: silu_layer
  implicit none

  type, extends(base_layer) :: swiglu_layer
    integer :: sequence_length, intermediate_size, model_dimension

    type(silu_layer) :: activation

    real, pointer :: gradient(:, :)
    real, pointer :: output(:, :)

    type(linear2d_layer) :: gate_proj
    type(linear2d_layer) :: up_proj
    type(linear2d_layer) :: down_proj

  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type swiglu_layer

  interface swiglu_layer
    module function swiglu_layer_cons(intermediate_size) result(res)
      integer, intent(in) :: intermediate_size
      type(swiglu_layer) :: res
    end function swiglu_layer_cons
  end interface swiglu_layer

contains
  module function swiglu_layer_cons(intermediate_size) result(res)
    integer, intent(in) :: intermediate_size
    type(swiglu_layer) :: res

    res % intermediate_size = intermediate_size
  end function swiglu_layer_cons

  subroutine forward(self, input)
    class(swiglu_layer), intent(inout), target :: self
    real :: input(:, :)
    integer :: i

    call self % gate_proj % forward(input)
    call self % up_proj % forward(input)

    do concurrent(i = 1: self % sequence_length)
      call self % activation % forward(self % gate_proj % output(i, :))
      self % gate_proj % output(i, :) = self % activation % output
    end do

    call self % down_proj % forward(self % gate_proj % output * self % up_proj % output)

    self % output => self % down_proj % output
  end subroutine forward

  subroutine backward(self, input, gradient)
    class(swiglu_layer), intent(inout) :: self
    real :: input(:, :)
    real :: gradient(:, :)

  end subroutine backward

  module subroutine init(self, input_shape)
    class(swiglu_layer), intent(inout) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "RMSNorm Layer accepts 2D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    self % gate_proj = linear2d_layer(self % intermediate_size, biases=.false.)
    self % up_proj = linear2d_layer(self % intermediate_size, biases=.false.)
    self % down_proj = linear2d_layer(self % model_dimension, biases=.false.)
    call self % gate_proj % init([self % sequence_length, self % model_dimension])
    call self % up_proj % init([self % sequence_length, self % model_dimension])
    call self % down_proj % init([self % sequence_length, self % intermediate_size])

    self % activation = silu_layer()
    call self % activation % init([self % intermediate_size])
  end subroutine init
end module llmf_swiglu
