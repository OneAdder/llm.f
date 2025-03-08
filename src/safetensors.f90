program read_binary_file
  use json_module, only: json_file
  implicit none
  integer(8) :: header_size
  character(len=:), allocatable :: json_data
  integer :: iunit, ios
  type(json_file) :: json

  open(unit=iunit, file='/home/misha/Проекты/llm.f/model.safetensors', form='unformatted', access='stream', status='old')

  ! Read the first 64 bits to get header_size
  read(iunit) header_size
  allocate(character(len=header_size) :: json_data)
  ! Read next `header_size` bits
  read(iunit) json_data

  call json % initialize()
  call json % deserialize(trim(adjustl(json_data)))
  call json % print()
!  print *, 'Read JSOheader_size:', trim(adjustl(json_data))

  ! Close the file
  close(iunit)

end program read_binary_file
