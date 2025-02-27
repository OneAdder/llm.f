#define ABSOLUTE_TOLERANCE 1e-06
#define RELATIVE_TOLERANCE 1e-05
#define _all_close(x, y) all(abs(x - y) <= (ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(y)))


module llmf_utils
  use iso_fortran_env, only: stderr => error_unit
  implicit none

  interface allclose
    procedure :: allclose_1d, allclose_2d, allclose_3d
  end interface allclose

contains
  function allclose_1d(x, y) result(res)
    real, intent(in) :: x(:)
    real, intent(in) :: y(:)
    logical :: res

    res = _all_close(x, y)
  end function allclose_1d

  function allclose_2d(x, y) result(res)
    real, intent(in) :: x(:, :)
    real, intent(in) :: y(:, :)
    logical :: res

    res = _all_close(x, y)
  end function allclose_2d

  function allclose_3d(x, y) result(res)
    real, intent(in) :: x(:, :, :)
    real, intent(in) :: y(:, :, :)
    logical :: res

    res = _all_close(x, y)
  end function allclose_3d

  subroutine assert_that(statement, ok, message)
    logical, intent(in) :: statement
    logical, intent(inout) :: ok
    character(len=*), intent(in) :: message

    if (.not. statement) then
      write(stderr, '(a)') message
      ok = .false.
    end if
  end subroutine assert_that
end module llmf_utils
