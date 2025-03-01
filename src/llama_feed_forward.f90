module llmf_llama_feed_forward
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer
  use llmf_silu, only: silu_layer
  implicit none

  type, extends(base_layer) :: llama_feed_forward_layer
    integer :: sequence_length, intermediate_size, model_dimension

    type(silu_layer) :: activation

    real, allocatable :: gradient(:, :)
    real, pointer :: output(:, :)

    type(linear2d_layer) :: gate_proj
    type(linear2d_layer) :: up_proj
    type(linear2d_layer) :: down_proj

    real, allocatable, private :: activation_output(:, :)
    real, allocatable, private :: hadamard_product_output(:, :)
    real, allocatable, private :: activation_gradient(:, :)

  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type llama_feed_forward_layer

  interface llama_feed_forward_layer
    module function llama_feed_forward_layer_cons(intermediate_size) result(res)
      integer, intent(in) :: intermediate_size
      type(llama_feed_forward_layer) :: res
    end function llama_feed_forward_layer_cons
  end interface llama_feed_forward_layer

contains
  module function llama_feed_forward_layer_cons(intermediate_size) result(res)
    integer, intent(in) :: intermediate_size
    type(llama_feed_forward_layer) :: res

    res % intermediate_size = intermediate_size
  end function llama_feed_forward_layer_cons

  subroutine forward(self, input)
    class(llama_feed_forward_layer), intent(inout), target :: self
    real :: input(:, :)

    call self % gate_proj % forward(input)
    call self % up_proj % forward(input)

    call self % activation % forward(self % gate_proj % output)
    self % activation_output = self % activation % output
    self % hadamard_product_output = self % activation % output * self % up_proj % output

    call self % down_proj % forward(self % hadamard_product_output)

    self % output => self % down_proj % output
  end subroutine forward

  subroutine backward(self, input, gradient)
    class(llama_feed_forward_layer), intent(inout), target :: self
    real :: input(:, :)
    real :: gradient(:, :)

    call self % down_proj % backward(self % hadamard_product_output, gradient)

    call self % activation % backward(&
        self % gate_proj % output,&
        self % down_proj % gradient * self % up_proj % output&
    )

    call self % gate_proj % backward(input, self % activation % gradient)
    call self % up_proj % backward(input, self % down_proj % gradient * self % activation_output)

    self % gradient = self % up_proj % gradient + self % gate_proj % gradient
  end subroutine backward

  module subroutine init(self, input_shape)
    class(llama_feed_forward_layer), intent(inout) :: self
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
    call self % activation % init([self % sequence_length, self % intermediate_size])

    allocate(self % gradient(self % sequence_length, self % model_dimension))

    allocate(self % activation_output(self % sequence_length, self % intermediate_size))
    allocate(self % hadamard_product_output(self % sequence_length, self % intermediate_size))
    allocate(self % activation_gradient(self % sequence_length, self % intermediate_size))
  end subroutine init
end module llmf_llama_feed_forward
