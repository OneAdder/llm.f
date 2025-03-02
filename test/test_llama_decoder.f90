program test_llama_decoder_layer
  use iso_fortran_env, only: stderr => error_unit
  use llmf_llama_decoder, only: llama_decoder_layer
  use llmf_utils, only: allclose, assert_that
  implicit none

  logical :: ok = .true.

  call test_decoder_qwen(ok)

  if (.not. ok) then
    write(stderr, '(a)') 'test_llama_decoder: one or more tests have failed'
  else
    print '(a)', 'test_llama_decoder: all tests have passed'
  end if

contains
  subroutine set_weights_qwen(decoder)
    type(llama_decoder_layer), intent(inout) :: decoder
    decoder % self_attn % query_layer % weights = 0.1
    decoder % self_attn % key_layer % weights = 0.2
    decoder % self_attn % value_layer % weights = 0.3
    decoder % self_attn % output_layer % weights = 0.2
    decoder % self_attn % query_layer % biases = 0.11
    decoder % self_attn % key_layer % biases = 0.11
    decoder % self_attn % value_layer % biases = 0.11

    decoder % feed_forward % gate_proj % weights = 0.01
    decoder % feed_forward % up_proj % weights = 0.05
    decoder % feed_forward % down_proj % weights = 0.1
  end subroutine set_weights_qwen

  subroutine test_decoder_qwen(ok)
    logical, intent(inout) :: ok
    type(llama_decoder_layer) :: decoder
    real :: input(9, 8) = reshape([&
      0.37072068, 1.3588632, 1.9884692, -0.30161408, 0.8139249, -0.8694511, -1.6688265, 1.1662576,&
      0.2468737, -0.9110192, 0.04235176, -0.45361742, 0.36150697, 0.8447159, 0.5774916, 0.08439375,&
      -0.6989709, -1.0674919, -1.1826763, -0.880081, 1.0227386, 0.2537887, 1.6236242, 1.6021129,&
      1.6466041, 1.0528897, -2.2763023, 1.2870983, 0.12475284, 0.13161187, -1.5640669, -0.14536288,&
      -0.5928059, -0.6204104, -1.8023925, 0.3605392, -0.47602218, -1.2931924, 1.0930126, -0.30460525,&
      -0.72116065, 1.3226725, -0.40716624, 0.67106473, 0.33057117, -0.2370891, 0.00500101, 1.2334017,&
      1.8437266, 0.14475724, 1.4683013, -0.17329404, -1.1277869, 0.5619295, 1.1053102, 1.0893365,&
      -0.13279338, 0.41584253, 1.8390387, -0.24315795, -0.71289355, 0.00285761, 0.29787052,&
      -1.5655739, 1.5732917, -0.17363702, 1.2494951, 0.10759168, -0.08730965, -1.1767657, -0.23054339, -0.97817636&
    ], [9, 8])
    real :: cosine(9, 2) = reshape([&
      1., 0.54030234, -0.41614684, -0.9899925, -0.6536436, 0.2836622, 0.96017027, 0.75390226, -0.14550003,&
      1., 0.54030234, -0.41614684, -0.9899925, -0.6536436, 0.2836622, 0.96017027, 0.75390226, -0.14550003&
    ], [9, 2])
    real :: sine(9, 2) = reshape([&
      0., 0.84147096, 0.9092974, 0.14112, -0.7568025, -0.9589243, -0.2794155, 0.6569866, 0.98935825,&
      0., 0.84147096, 0.9092974, 0.14112, -0.7568025, -0.9589243, -0.2794155, 0.6569866, 0.98935825&
    ], [9, 2])
    real :: gradient(9, 8) = reshape([&
      0.2643, 0.5053, 0.8736, 0.7107, 0.4371, 0.7562, 0.4131, 0.2426, 0.3533,&
      0.4271, 0.8181, 0.9274, 0.3118, 0.4004, 0.7692, 0.8253, 0.8845, 0.5925,&
      0.8704, 0.5688, 0.6746, 0.9214, 0.3114, 0.9759, 0.6638, 0.0278, 0.3023,&
      0.7629, 0.1311, 0.7574, 0.9603, 0.6012, 0.3955, 0.476, 0.9791, 0.9799,&
      0.5051, 0.2189, 0.426, 0.431, 0.3824, 0.7647, 0.7132, 0.1745, 0.3256,&
      0.6396, 0.1247, 0.9805, 0.7546, 0.5645, 0.9313, 0.4169, 0.7757, 0.0203,&
      0.5826, 0.6509, 0.1613, 0.426, 0.8945, 0.3592, 0.5082, 0.8625, 0.3525,&
      0.76, 0.9715, 0.5208, 0.979, 0.1589, 0.6575, 0.1034, 0.0686, 0.5525&
    ], [9, 8])

    real :: expected_output(9, 8) = reshape([&
      -0.20685023, 1.7706567, 3.43436, 0.81435204, 2.2747076, 0.5662979, -0.6733629, 2.0058637, 0.8646039,&
      -1.4885503, 0.45415676, 0.99226004, 1.477452, 2.3055074, 2.0132978, 1.0798371, 0.1405636, -0.44979614,&
      -1.7602502, -0.46834326, 2.46856, 1.369752, 3.0844076, 3.0378978, 2.6420372, 1.8924637, -1.6585962,&
      0.7095497, 0.5365568, 1.57746, -0.448148, 1.3154074, 0.84299785, 0.37503707, -0.9628364, 0.9782039,&
      -1.0535502, -0.8814432, 2.53886, 0.811352, 0.7396075, 2.758498, 0.58823705, 1.5106637, 0.9483038,&
      -0.81465024, 0.41675675, 2.67926, 2.9596522, 1.6056075, 2.9040978, 0.8221371, -0.28823638, 1.1796039,&
      0.5277497, 1.5010568, 1.31306, 1.531752, 3.2998073, 1.1925979, 0.2825371, 0.8424636, 0.9156039,&
      -2.14315, 1.9850568, 1.2722601, 2.3654523, 1.5684074, 1.3484979, -0.18136293, 0.6090636, -0.36049616&
    ], [9, 8])
    real :: expected_gradient(9, 8) = reshape([&
      6.94482, 3.0779157, 0.15215492, 3.2654061, 1.7998018, 2.89182, 0.7450665, 0.72557074, 0.6070686,&
      5.513385, 4.713659, 6.1422367, 2.4708018, 1.718844, 1.9802911, 1.6712003, 1.2945789, 0.7577533,&
      5.618732, 5.391374, 2.3008318, 3.1446717, 0.5106056, 1.5323453, 1.9676443, 0.5063393, 0.38614944,&
      8.583279, 3.9438558, 4.549727, 4.268402, 3.3423657, 2.3543286, 1.1152949, 1.3460605, 1.2413188,&
      6.132457, 5.4565954, 1.8813455, 2.9874964, 3.9509602, 1.4996616, 1.4149925, 0.63811946, 0.58500516,&
      6.564111, 4.0578423, 2.09456, 2.0291, 2.8886635, 1.5732338, 1.187258, 1.1690223, 0.2952816,&
      8.176847, 3.4944353, 4.596333, 2.5525982, 0.78418684, 2.0946593, 1.1203793, 1.3000076, 0.60970306,&
      5.0320644, 3.3286667, 5.0550103, 2.6080885, 2.536518, 2.2933505, 0.5795916, 0.49698687, 0.72376704&
    ], [9, 8])

    decoder = llama_decoder_layer(intermediate_size=32, n_heads=4, n_kv_heads=2, is_qwen=.true.)
    call decoder % init([9, 8])
    call set_weights_qwen(decoder)

    call decoder % forward(input, cosine, sine)
    call assert_that(&
        allclose(decoder % output, expected_output), ok,&
        'incorrect output after forward pass (qwen)'&
    )

!    call decoder % backward(input, gradient, cosine, sine)
!
!    call assert_that(&
!        allclose(decoder % gradient, expected_gradient), ok, 'incorrect gradient after backward pass (qwen)'&
!    )
  end subroutine test_decoder_qwen
end program test_llama_decoder_layer
