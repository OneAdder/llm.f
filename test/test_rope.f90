program test_rope
  use iso_fortran_env, only: stderr => error_unit
  use llmf_rope, only: rotary_embedding
  use llmf_utils, only: allclose, assert_that
  implicit none

  logical :: ok = .true.

  call test_numeric(ok)
  call test_shapes(ok)

  if (.not. ok) then
    write(stderr, '(a)') 'test_rope: one or more tests have failed'
    error stop
  else
    print '(a)', 'test_rope: all tests have passed'
  end if
contains
  subroutine test_numeric(ok)
    logical, intent(inout) :: ok

    integer :: position_ids(9) = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    real :: expected_inv_freq(4) = [1., 0.1, 0.01, 0.001]
    real :: expected_cos(9, 8) = reshape([&
      1., 0.54030234, -0.41614684, -0.9899925, -0.6536436, 0.2836622, 0.96017027, 0.75390226, -0.14550003,&
      1., 0.9950042, 0.9800666, 0.9553365, 0.921061, 0.87758255, 0.8253356, 0.7648422, 0.6967067,&
      1., 0.99995, 0.9998, 0.99955004, 0.9992001, 0.99875027, 0.99820054, 0.997551, 0.9968017,&
      1., 0.9999995, 0.99999803, 0.9999955, 0.999992, 0.9999875, 0.999982, 0.9999755, 0.999968,&
      1., 0.54030234, -0.41614684, -0.9899925, -0.6536436, 0.2836622, 0.96017027, 0.75390226, -0.14550003,&
      1., 0.9950042, 0.9800666, 0.9553365, 0.921061, 0.87758255, 0.8253356, 0.7648422, 0.6967067,&
      1., 0.99995, 0.9998, 0.99955004, 0.9992001, 0.99875027, 0.99820054, 0.997551, 0.9968017,&
      1., 0.9999995, 0.99999803, 0.9999955, 0.999992, 0.9999875, 0.999982, 0.9999755, 0.999968&
    ], [9, 8])
    real :: expected_sin(9, 8) = reshape([&
      0., 0.84147096, 0.9092974, 0.14112, -0.7568025, -0.9589243, -0.2794155, 0.6569866, 0.98935825,&
      0., 0.09983342, 0.19866933, 0.29552022, 0.38941836, 0.47942555, 0.5646425, 0.64421767, 0.7173561,&
      0., 0.00999983, 0.01999867, 0.0299955, 0.03998933, 0.04997917, 0.059964, 0.06994285, 0.07991469,&
      0., 0.001, 0.002, 0.003, 0.00399999, 0.00499998, 0.00599996, 0.00699994, 0.00799991,&
      0., 0.84147096, 0.9092974, 0.14112, -0.7568025, -0.9589243, -0.2794155, 0.6569866, 0.98935825,&
      0., 0.09983342, 0.19866933, 0.29552022, 0.38941836, 0.47942555, 0.5646425, 0.64421767, 0.7173561,&
      0., 0.00999983, 0.01999867, 0.0299955, 0.03998933, 0.04997917, 0.059964, 0.06994285, 0.07991469,&
      0., 0.001, 0.002, 0.003, 0.00399999, 0.00499998, 0.00599996, 0.00699994, 0.00799991&
    ], [9, 8])

    type(rotary_embedding) :: rope
    real :: cosine(9, 8)
    real :: sine(9, 8)

    rope = rotary_embedding(sequence_length=9, model_dimension=8, head_size=8)

    call assert_that(allclose(rope % inv_freq, expected_inv_freq), ok, 'constructor calculated incorrect inv_freq')

    call rope % apply(position_ids, cosine, sine)

    call assert_that(allclose(cosine, expected_cos), ok, 'cos was calculated incorrectly')
    call assert_that(allclose(sine, expected_sin), ok, 'sin was calculated incorrectly')
  end subroutine test_numeric

  subroutine test_shapes(ok)
    logical, intent(inout) :: ok

    integer :: position_ids(9) = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    integer :: expected_inv_freq_shape(1) = [32]
    integer :: expected_cos_shape(2) = [9, 64]
    integer :: expected_sin_shape(2) = [9, 64]

    type(rotary_embedding) :: rope
    real :: cosine(9, 64)
    real :: sine(9, 64)

    rope = rotary_embedding(sequence_length=9, model_dimension=896, head_size=64)

    call assert_that(&
        all(shape(rope % inv_freq) == expected_inv_freq_shape), ok,&
        'constructor made incorrect shape for inv_freq'&
    )

    call rope % apply(position_ids, cosine, sine)

    call assert_that(all(shape(cosine) == expected_cos_shape), ok, 'cos has incorrect shape')
    call assert_that(all(shape(sine) == expected_sin_shape), ok, 'sin has incorrect shape')
  end subroutine test_shapes
end program test_rope
