module llmf_llama_attention
  use nf_multihead_attention_layer, only: multihead_attention_layer
  implicit none

  private
  public :: llama_attention_layer

  type, extends(multihead_attention_layer) :: llama_attention_layer
  end type llama_attention_layer

  interface llama_attention_layer
    module function llama_attention_layer_cons(n_heads) result(res)
      integer, intent(in) :: n_heads
      type(llama_attention_layer) :: res
    end function llama_attention_layer_cons
  end interface llama_attention_layer
end module llmf_llama_attention
