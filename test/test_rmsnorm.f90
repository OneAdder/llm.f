program test_rmsnorm
  use iso_fortran_env, only: stderr => error_unit
  use llmf_rmsnorm, only: rmsnorm_layer
  use llmf_utils, only: allclose, assert_that
  implicit none

  logical :: ok = .true.

  call test_forward(ok)

  if (.not. ok) then
    write(stderr, '(a)') 'test_rmsnorm: one or more tests have failed'
  end if
contains
  subroutine test_forward(ok)
    logical, intent(inout) :: ok

    real :: input(2, 3) = reshape([&
      -0.5, 0.1, 12.,&
      -13., 0.6, -0.7&
    ], [2, 3])
    real :: gradient(2, 3) = reshape([0.1, 3.1, 2.7, 3., 1.4, 0.2], [2, 3])
    real :: expected_output(2, 3) = reshape([&
        -0.07201641, 0.0133038, 1.7283938,&
        -1.7294942, 0.08641969, -0.09312661&
    ], [2, 3])
    real :: expected_gradient(2, 3) = reshape([&
        0.03093205, 0.41546553, -0.00780177,&
        0.00291198, 0.18181142, 0.00527365&
    ], [2, 3])
    real :: expected_dw(3) = [0.03404, -0.5218, 0.1023622]

    type(rmsnorm_layer) :: rmsnorm

    rmsnorm = rmsnorm_layer()
    call rmsnorm % init([2, 3])

    call rmsnorm % forward(input)
    call assert_that(allclose(rmsnorm % output, expected_output), ok, 'forward returned incorrect values')

    call rmsnorm % backward(input, gradient)
    call assert_that(allclose(rmsnorm % gradient, expected_gradient), ok, 'backward returned incorrect gradient')
    call assert_that(allclose(rmsnorm % dw, expected_dw), ok, 'backward returned incorrect dw')
  end subroutine test_forward
end program test_rmsnorm