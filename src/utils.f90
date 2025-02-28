#define ABSOLUTE_TOLERANCE 1e-06
#define RELATIVE_TOLERANCE 1e-05
#define _all_close(x, y) all(abs(x - y) <= (ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * abs(y)))


module llmf_utils
  use iso_fortran_env, only: stderr => error_unit
  implicit none

  private
  public allclose, assert_that, init_kaiming

  interface allclose
    procedure :: allclose_1d, allclose_2d, allclose_3d
  end interface allclose

  interface init_kaiming
    procedure :: init_kaiming_1d, init_kaiming_2d
  end interface init_kaiming

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

  subroutine init_kaiming_1d(x, n_prev)
    !! Kaiming weight initialization
    real, intent(inout) :: x(:)
    integer, intent(in) :: n_prev
    real :: stdv
    call random_number(x)
    x = x * sqrt(2. / n_prev)
  end subroutine init_kaiming_1d

  subroutine init_kaiming_2d(x, n_prev)
    !! Kaiming weight initialization
    real, intent(inout) :: x(:, :)
    integer, intent(in) :: n_prev
    real :: stdv
    call random_number(x)
    x = x * sqrt(2. / n_prev)
  end subroutine init_kaiming_2d
end module llmf_utils
