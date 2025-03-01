program test_rmsnorm
  use iso_fortran_env, only: stderr => error_unit
  use llmf_silu, only: silu_layer
  use llmf_utils, only: allclose, assert_that
  implicit none

  logical :: ok = .true.

  call test_silu(ok)

  if (.not. ok) then
    write(stderr, '(a)') 'test_swiglu: one or more tests have failed'
  end if

contains
  subroutine test_silu(ok)
    logical, intent(inout) :: ok

    real :: input(4) = [-3., -0.5, 0.1, 15.]
    real :: gradient(4) = [1., 2., 0.5, 3.]
    real :: expected_output(4) = [-0.142278, -0.18877, 0.0524979, 15.]
    real :: expected_gradient(4) = [-0.0881, 0.5201, 0.274958402, 3.]
    type(silu_layer) :: silu

    silu = silu_layer()

    call silu % forward(input)
    call assert_that(allclose(silu % output, expected_output), ok, 'silu was calculated incorrectly')

    call silu % backward(input, gradient)
    call assert_that(allclose(silu % gradient, expected_gradient), ok, 'silu was calculated incorrectly')
  end subroutine test_silu
end program test_rmsnorm
