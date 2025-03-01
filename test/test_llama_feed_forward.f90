program test_llama_feed_forward
  use iso_fortran_env, only: stderr => error_unit
  use llmf_silu, only: silu_layer
  use llmf_llama_feed_forward, only: llama_feed_forward_layer
  use llmf_utils, only: allclose, assert_that
  implicit none

  logical :: ok = .true.

  call test_silu(ok)
  call test_llama_feed_forward_layer(ok)

  if (.not. ok) then
    write(stderr, '(a)') 'test_llama_feed_forward: one or more tests have failed'
  end if

contains
  subroutine test_silu(ok)
    logical, intent(inout) :: ok

    real :: input(1, 4) = reshape([-3., -0.5, 0.1, 15.], [1, 4])
    real :: gradient(1, 4) = reshape([1., 2., 0.5, 3.], [1, 4])
    real :: expected_output(1, 4) = reshape([-0.142278, -0.18877, 0.0524979, 15.], [1, 4])
    real :: expected_gradient(1, 4) = reshape([-0.0881, 0.5201, 0.274958402, 3.], [1, 4])
    type(silu_layer) :: silu

    silu = silu_layer()
    call silu % init([1, 4])

    call silu % forward(input)
    call assert_that(allclose(silu % output, expected_output), ok, 'silu was calculated incorrectly')

    call silu % backward(input, gradient)
    call assert_that(allclose(silu % gradient, expected_gradient), ok, 'silu was calculated incorrectly')
  end subroutine test_silu

  subroutine test_llama_feed_forward_layer(ok)
    logical, intent(inout) :: ok

    real :: input(2, 3) = reshape([&
        -0.9812, -2.,&
        -1.0309, -0.9520,&
        1.0083, -1.0007&
    ], [2, 3])
    real :: gradient(2, 3) = reshape([&
        0.1, 3.,&
        2., 0.3,&
        3.1, 0.1&
    ], [2, 3])
    real :: expected_output(2, 3) = reshape([&
        0.04221962, 0.36401537,&
        0.07036602, 0.60669225,&
        0.09851244, 0.84936917&
    ], [2, 3])
    real :: expected_gradient(2, 3) = reshape([&
        -0.81758773, -0.39875072,&
        -0.81758773, -0.39875072,&
        -0.81758773, -0.39875072&
    ], [2, 3])
    real :: expected_gate_proj_dw(3, 4) = reshape([&
        1.4488734, 0.9748564, -0.03336787,&
        0.75658035, 0.57763064, -0.1997615,&
        0.3245887, 0.3137403, -0.2609922,&
        0.19577153, 0.31379288, -0.48862416&
    ], [3, 4])
    real :: expected_up_proj_dw(3, 4) = reshape([&
        0.2530082, 0.16344056, 0.01223498,&
        0.41811574, 0.28052184, -0.00749654,&
        0.51184654, 0.359177, -0.0511069,&
        0.55491436, 0.40937614, -0.10852566&
    ], [3, 4])
    real :: expected_down_proj_dw(4, 3) = reshape([&
        1.3237882, 1.7606071, 1.6516823, 2.572376,&
        0.19903979, 0.28433365, 0.29308063, 0.5145049,&
        0.14785829, 0.22717193, 0.25410232, 0.48608306&
    ], [4, 3])
    type(llama_feed_forward_layer) :: feed_forward

    feed_forward = llama_feed_forward_layer(4)
    call feed_forward % init([2, 3])
    feed_forward % gate_proj % weights = reshape([&
        0.1, 0.1, 0.1,&
        0.2, 0.2, 0.2,&
        0.3, 0.3, 0.3,&
        0.4, 0.4, 0.4&
    ], [3, 4])
    feed_forward % up_proj % weights = reshape([&
        0.7, 0.7, 0.7,&
        0.6, 0.6, 0.6,&
        0.5, 0.5, 0.5,&
        0.8, 0.8, 0.8&
    ], [3, 4])
    feed_forward % down_proj % weights = reshape([&
        0.15, 0.15, 0.15, 0.15,&
        0.25, 0.25, 0.25, 0.25,&
        0.35, 0.35, 0.35, 0.35&
    ], [4, 3])

    call feed_forward % forward(input)
    call assert_that(&
        allclose(feed_forward % output, expected_output), ok,&
        'llama_feed_forward forward: incorrect values'&
    )

    call feed_forward % backward(input, gradient)
    call assert_that(&
        allclose(feed_forward % gradient, expected_gradient), ok,&
        'llama_feed_forward backward: incorrect gradient'&
    )
    call assert_that(&
        allclose(feed_forward % gate_proj % dw, expected_gate_proj_dw), ok,&
        'llama_feed_forward backward: incorrect `gate_proj` weights gradient'&
    )
    call assert_that(&
        allclose(feed_forward % up_proj % dw, expected_up_proj_dw), ok,&
        'llama_feed_forward backward: incorrect `up_proj` weights gradient'&
    )
    call assert_that(&
        allclose(feed_forward % down_proj % dw, expected_down_proj_dw), ok,&
        'llama_feed_forward backward: incorrect `down_proj` weights gradient'&
    )
  end subroutine test_llama_feed_forward_layer
end program test_llama_feed_forward
