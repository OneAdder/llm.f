module llmf_qwen2_attention
  use nf_multihead_attention_layer, only: multihead_attention_layer
  implicit none

  private
  public :: qwen2_attention_layer

  type, extends(multihead_attention_layer) :: qwen2_attention_layer
  end type qwen2_attention_layer

  interface qwen2_attention_layer
    module function qwen2_attention_layer_cons(n_heads) result(res)
      integer, intent(in) :: n_heads
      type(qwen2_attention_layer) :: res
    end function qwen2_attention_layer_cons
  end interface qwen2_attention_layer
end module llmf_qwen2_attention
