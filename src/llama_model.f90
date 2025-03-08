!! Python Reference: https://github.com/OneAdder/neural-fortran-references/blob/main/qwen_model.py
#define BRACKET_NOTATION 3
#define LAYER_KEY_MAXLEN 100

module llmf_llama_model
  use iso_fortran_env, only: stderr => error_unit
  use json_module, only: json_file
  use nf_base_layer, only: base_layer
  use nf_embedding_layer, only: embedding_layer
  use nf_linear2d_layer, only: linear2d_layer
  use llmf_rope, only: rotary_embedding
  use llmf_llama_decoder, only: llama_decoder_layer
  use llmf_rmsnorm, only: rmsnorm_layer
  implicit none

  type, extends(base_layer) :: llama_model
    integer :: batch_size, sequence_length, intermediate_size, model_dimension
    integer :: vocab_size, n_layers, n_heads, n_kv_heads
    logical :: is_qwen

    real, allocatable :: gradient(:, :, :)
    real, pointer :: output(:, :, :)

    real, allocatable :: last_hidden_state(:, :, :)

    type(embedding_layer) :: embed_tokens
    type(rotary_embedding) :: rope
    type(llama_decoder_layer), allocatable :: decoder_stack(:)
    type(rmsnorm_layer) :: norm
    type(linear2d_layer) :: lm_head

  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
    procedure :: init_pretrained
    procedure :: read_safetensors
  end type llama_model

  interface llama_model
    module function llama_model_cons(&
        vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size, is_qwen&
    ) result(res)
      integer, intent(in) :: vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size
      logical, optional :: is_qwen
      type(llama_model) :: res
    end function llama_model_cons
  end interface llama_model

contains
  module function llama_model_cons(&
      vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size, is_qwen&
  ) result(res)
    integer, intent(in) :: vocab_size, n_layers, n_heads, n_kv_heads, intermediate_size
    logical, optional :: is_qwen
    type(llama_model) :: res

    res % vocab_size = vocab_size
    res % n_layers = n_layers
    res % n_heads = n_heads
    res % n_kv_heads = n_kv_heads
    res % intermediate_size = intermediate_size
    if (present(is_qwen)) then
      res % is_qwen = is_qwen
    else
      res % is_qwen = .false.
    end if
  end function llama_model_cons

  subroutine forward(self, input, attention_mask)
    class(llama_model), intent(inout), target :: self
    integer :: input(:, :)
    logical :: attention_mask(:, :)
    integer :: batch, pos, layer, i, j

    integer :: position_ids(self % sequence_length)
    real :: causal_attention_mask(self % sequence_length, self % sequence_length, self % batch_size)
    real :: embedding_output(self % sequence_length, self % model_dimension, self % batch_size)
    real :: cosine(self % sequence_length, self % rope % head_size)
    real :: sine(self % sequence_length, self % rope % head_size)
    real :: decoder_stack_output(self % sequence_length, self % model_dimension, self % batch_size)

    do batch = 1, self % batch_size
      position_ids = [(pos, pos=0, self % sequence_length, 1)]

      causal_attention_mask = 0.
      do concurrent(i = 1: self % sequence_length, j = 1: self % sequence_length)
        if (i < j .or. .not. attention_mask(j, batch)) then
          causal_attention_mask(i, j, batch) = -100.
        else
          causal_attention_mask(i, j, batch) = 0.
        end if
      end do

      call self % embed_tokens % forward(input(:, batch))
      embedding_output(:, :, batch) = self % embed_tokens % output

      call self % rope % apply(position_ids, cosine, sine)
      call self % decoder_stack(1) % forward(&
          embedding_output(:, :, batch), cosine, sine, causal_attention_mask(:, :, batch)&
      )
      decoder_stack_output(:, :, batch) = self % decoder_stack(1) % output
      do layer = 2, self % n_layers
        call self % decoder_stack(layer) % forward(&
            decoder_stack_output(:, :, batch), cosine, sine, causal_attention_mask(:, :, batch)&
        )
        decoder_stack_output(:, :, batch) = self % decoder_stack(layer) % output
      end do
      call self % norm % forward(self % decoder_stack(self % n_layers) % output)
      self % last_hidden_state(:, :, batch) = self % norm % output

      call self % lm_head % forward(self % norm % output)
      self % output(:, :, batch) = self % lm_head % output
    end do
  end subroutine forward

  subroutine backward(self, input, gradient, attention_mask)
    class(llama_model), intent(inout), target :: self
    integer :: input(:, :)
    real :: gradient(:, :)
    logical :: attention_mask(:, :)

  end subroutine backward

  module subroutine init(self, input_shape)
    class(llama_model), intent(inout) :: self
    integer, intent(in) :: input_shape(:)
    integer :: i

    if (size(input_shape) /= 3) then
      error stop "Llama Model accepts 3D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)
    self % batch_size = input_shape(3)

    ! init layers
    self % embed_tokens = embedding_layer(self % vocab_size, self % model_dimension)
    call self % embed_tokens % init([self % sequence_length])

    self % rope = rotary_embedding(&
        self % sequence_length, self % model_dimension, self % model_dimension / self % n_heads&
    )

    allocate(self % decoder_stack(self % n_layers))
    do i = 1, self % n_layers
      self % decoder_stack(i) = llama_decoder_layer(&
          intermediate_size=self % intermediate_size,&
          n_heads=4,&
          n_kv_heads=2,&
          is_qwen=self % is_qwen&
      )
      call self % decoder_stack(i) % init([self % sequence_length, self % model_dimension])
    end do

    self % norm = rmsnorm_layer()
    call self % norm % init([self % sequence_length, self % model_dimension])

    self % lm_head = linear2d_layer(self % vocab_size, biases=.false.)
    call self % lm_head % init([self % sequence_length, self % model_dimension])

    ! allocate public
    allocate(self % output(self % sequence_length, self % vocab_size, self % batch_size))
    allocate(self % gradient(self % sequence_length, self % model_dimension, self % batch_size))

    allocate(self % last_hidden_state(self % sequence_length, self % model_dimension, self % batch_size))
  end subroutine init

  module subroutine init_pretrained(self, safetensors_path)
    class(llama_model), intent(inout) :: self
    character(len=*), intent(in) :: safetensors_path

    integer(8) :: header_size
    character(len=:), allocatable :: header
    integer iunit, ios
    type(json_file) :: json
    logical :: is_present

    integer :: start_pos
    integer :: layer
    character(len=LAYER_KEY_MAXLEN) :: layer_key

    open(unit=iunit, file=safetensors_path, form='unformatted', access='stream', status='old')

    ! Read the first 64 bits to get header_size and then JSON header itself
    read(iunit) header_size
    allocate(character(len=header_size) :: header)
    read(iunit) header
    inquire(unit=iunit, pos=start_pos)

    ! parse JSON
    call json % initialize(path_mode=BRACKET_NOTATION)
    call json % deserialize(trim(adjustl(header)))

    ! read data from safetensors
    ! FIXME: at this point we only work with 1D and 2D shapes
    self % lm_head % weights = self % read_safetensors('lm_head.weight', json, iunit, start_pos)
    self % embed_tokens % weights = transpose(self % read_safetensors('model.embed_tokens.weight', json, iunit, start_pos))
    do layer = 1, self % n_layers
      write(layer_key, '(A,I0)') 'model.layers.', layer - 1
      self % decoder_stack(layer) % self_attn % query_layer % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.self_attn.q_proj.weight', json, iunit, start_pos&
          )
      self % decoder_stack(layer) % self_attn % key_layer % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.self_attn.k_proj.weight', json, iunit, start_pos&
          )
      self % decoder_stack(layer) % self_attn % value_layer % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.self_attn.v_proj.weight', json, iunit, start_pos&
          )
      self % decoder_stack(layer) % self_attn % output_layer % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.self_attn.o_proj.weight', json, iunit, start_pos&
          )
      self % decoder_stack(layer) % post_attention_layernorm % gamma = pack(&
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.post_attention_layernorm.weight', json, iunit, start_pos&
          ),&
          .true.&
      )
      self % decoder_stack(layer) % feed_forward % gate_proj % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.mlp.gate_proj.weight', json, iunit, start_pos&
          )
      self % decoder_stack(layer) % feed_forward % up_proj % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.mlp.up_proj.weight', json, iunit, start_pos&
          )
      self % decoder_stack(layer) % feed_forward % down_proj % weights = &
          self % read_safetensors(&
              trim(adjustl(layer_key)) // '.mlp.down_proj.weight', json, iunit, start_pos&
          )
      if (self % is_qwen) then
        self % decoder_stack(layer) % self_attn % query_layer % biases = pack(&
            self % read_safetensors(&
                trim(adjustl(layer_key)) // '.self_attn.q_proj.bias', json, iunit, start_pos&
            ),&
            .true.&
        )
        self % decoder_stack(layer) % self_attn % key_layer % biases = pack(&
            self % read_safetensors(&
                trim(adjustl(layer_key)) // '.self_attn.k_proj.bias', json, iunit, start_pos&
            ),&
            .true.&
        )
        self % decoder_stack(layer) % self_attn % value_layer % biases = pack(&
            self % read_safetensors(&
                trim(adjustl(layer_key)) // '.self_attn.v_proj.bias', json, iunit, start_pos&
            ),&
            .true.&
        )
      end if
    end do
    self % norm % gamma = pack(self % read_safetensors('model.norm.weight', json, iunit, start_pos), .true.)
  end subroutine init_pretrained

  module function read_safetensors(self, key, json_header, n_unit, start_pos) result(res)
    class(llama_model), intent(inout) :: self
    character(len=*) :: key
    type(json_file), intent(inout) :: json_header
    integer, intent(in) :: n_unit
    integer, intent(in) :: start_pos
    real, allocatable :: res(:, :)

    logical :: is_present
    integer :: shape_first
    integer :: shape_second
    integer :: offset_start
    integer :: offset_end

    call json_header % get("$['" // key // "']['shape'][1]", shape_first, is_present)
    if (.not. is_present) then
      print *, 'Failed to find key ' // key
      error stop 1
    end if
    call json_header % get("$['" // key // "']['shape'][2]", shape_second, is_present)
    if (.not. is_present) shape_second = 1
    call json_header % get("$['" // key // "']['data_offsets'][1]", offset_start, is_present)
    call json_header % get("$['" // key // "']['data_offsets'][2]", offset_end, is_present)

    allocate(res(shape_second, shape_first))
    read(n_unit, pos=start_pos + offset_start) res
    rewind(n_unit)
  end function read_safetensors
end module llmf_llama_model
