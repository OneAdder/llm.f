program test_llama_model
  use iso_fortran_env, only: stderr => error_unit
  use llmf_llama_model, only: llama_model
  use llmf_utils, only: allclose, assert_that
  implicit none

  logical :: ok = .true.

  call test_llama_qwen(ok)

  if (.not. ok) then
    write(stderr, '(a)') 'test_llama: one or more tests have failed'
  else
    print '(a)', 'test_llama: all tests have passed'
  end if

contains
  subroutine set_weights_qwen(llama)
    type(llama_model), intent(inout) :: llama

    llama % embed_tokens % weights = spread(&
        [-2., -0.223, 0.3, 3., -0.1, 1.28, 2.28, -2.28],&
        dim=1, ncopies=llama % vocab_size&
    )

    llama % decoder_stack(1) % self_attn % query_layer % weights = 0.1
    llama % decoder_stack(1) % self_attn % key_layer % weights = 0.2
    llama % decoder_stack(1) % self_attn % value_layer % weights = 0.3
    llama % decoder_stack(1) % self_attn % output_layer % weights = 0.2
    llama % decoder_stack(1) % self_attn % query_layer % biases = 0.11
    llama % decoder_stack(1) % self_attn % key_layer % biases = 0.11
    llama % decoder_stack(1) % self_attn % value_layer % biases = 0.11
    llama % decoder_stack(1) % feed_forward % gate_proj % weights = 0.01
    llama % decoder_stack(1) % feed_forward % up_proj % weights = 0.05
    llama % decoder_stack(1) % feed_forward % down_proj % weights = 0.1

    llama % decoder_stack(2) % self_attn % query_layer % weights = 0.2
    llama % decoder_stack(2) % self_attn % key_layer % weights = 0.1
    llama % decoder_stack(2) % self_attn % value_layer % weights = 0.3
    llama % decoder_stack(2) % self_attn % output_layer % weights = 0.1
    llama % decoder_stack(2) % self_attn % query_layer % biases = 0.11
    llama % decoder_stack(2) % self_attn % key_layer % biases = 0.11
    llama % decoder_stack(2) % self_attn % value_layer % biases = 0.11
    llama % decoder_stack(2) % feed_forward % gate_proj % weights = 0.02
    llama % decoder_stack(2) % feed_forward % up_proj % weights = 0.06
    llama % decoder_stack(2) % feed_forward % down_proj % weights = 0.2
  end subroutine set_weights_qwen

  subroutine test_llama_qwen(ok)
    logical, intent(inout) :: ok
    type(llama_model) :: llama

    integer :: input(9, 1) = reshape([641, 9881, 358, 653, 537, 2948, 39244, 448, 10485], [9, 1])
    logical :: attention_mask(9, 1) = reshape([&
      .true., .true., .true., .true., .true., .true., .true., .true., .true.&
    ], [9, 1])

    real :: expected_output(9, 8, 1) = reshape([&
      0.01950932, 0.01950932, 0.01950932, 0.01950936, 0.01950932, 0.01950936, 0.01950936, 0.01950936, 0.01950932,&
      0.62758315, 0.6275831, 0.6275831, 0.62758315, 0.6275831, 0.6275831, 0.6275831, 0.6275831, 0.6275831,&
      0.8065491, 0.8065491, 0.8065491, 0.80654913, 0.8065491, 0.8065491, 0.8065491, 0.8065491, 0.8065491,&
      1.7304654, 1.7304654, 1.7304654, 1.7304653, 1.7304653, 1.7304653, 1.7304653, 1.7304653, 1.7304653,&
      0.6696726, 0.6696726, 0.6696726, 0.66967267, 0.66967255, 0.66967255, 0.6696726, 0.6696726, 0.66967255,&
      1.1418965, 1.1418965, 1.1418965, 1.1418965, 1.1418964, 1.1418965, 1.1418965, 1.1418965, 1.1418964,&
      1.4840877, 1.4840877, 1.4840877, 1.4840876, 1.4840876, 1.4840876, 1.4840876, 1.4840876, 1.4840876,&
      -0.0763042, -0.07630421, -0.07630421, -0.07630416, -0.0763042, -0.07630416, -0.07630416, -0.07630416, -0.0763042&
    ], [9, 8, 1])

    llama = llama_model(vocab_size=151643, n_layers=2, intermediate_size=32, n_heads=4, n_kv_heads=2, is_qwen=.true.)
    call llama % init([9, 8, 1])
    call set_weights_qwen(llama)

    call llama % forward(input, attention_mask)
    call assert_that(&
        allclose(llama % last_hidden_state, expected_output), ok,&
        'incorrect output after forward pass (qwen)'&
    )
!
!    call llama % backward(input, gradient, cosine, sine)
!    call assert_that(&
!        allclose(llama % gradient, expected_gradient), ok, 'incorrect gradient after backward pass (qwen)'&
!    )
  end subroutine test_llama_qwen
end program test_llama_model
