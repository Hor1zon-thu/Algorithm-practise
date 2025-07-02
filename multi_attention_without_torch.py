import numpy as np

#不使用torch库编写
class multi_attention():
    def __init__(self,emd_size,q_k_size,v_size,head_num,dropout=0.0,bias=None):
        super(multi_attention,self).__init__()
        self.emd_size =emd_size
        self.q_k_size = q_k_size
        self.v_size = v_size
        self.head_num = head_num
        self.head_dim = emd_size//head_num
        self.dropout = dropout

        assert self.head_num*self.head_dim == self.emd_size
        #emd_size should be divisible by numheads
        self.Wq =np.random.randn(emd_size,q_k_size*head_num)*0.01
        #缩小初始化权重，避免梯度爆炸
        self.Wk = np.random.randn(emd_size,q_k_size*head_num)*0.01
        self.Wv = np.random.randn(emd_size,v_size*head_num)*0.01
        self.out_proj = np.random.randn(v_size*head_num,emd_size)

        if bias :
            self.q_bias = np.zeros(emd_size)
            self.k_bias = np.zeros(emd_size)
            self.v_bias = np.zeros(emd_size)
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

    def softmax(self,x,axis=-1):
        x_max = np.max(x,axis=axis,keepdims=True)
        #除去最大值，保持稳定性。保持x维度不变
        x = x-x_max
        exp_x = np.exp(x)
        return exp_x /np.sum(exp_x,axis=axis,keepdims=True)

    def Dropout(self,x):
        if self.dropout == 0:
            return x
        else:
            mask = np.random.rand(*x.shape) > self.dropout
            # mask返回一个随机布尔量的矩阵
            x = x*mask /(1-self.dropout)
            #进行缩放，保证训练和输出期望一致
            return x



    def forward(self,x_q,x_k_v,mask=None):  #x为(batch_size,q_len,emd_size)
        batch_size = x_q.shape[0]
        q_len = x_q.shape[1]
        k_v_len = x_k_v.shape[1]
        #nn.Linear 是类实例，使用__call__方法，可以直接调用，如q= self.Wq(x_q)
        q = np.dot(x_q,self.Wq)  # [batch_size, q_len, emd_size]
        k = np.dot(x_k_v,self.Wk)
        v = np.dot(x_k_v,self.Wv)

        if self.q_bias is not None:
            q = q + self.q_bias
            k = k + self.k_bias
            v = v + self.v_bias
#numpy中的transpose方法与torch不同，transpose(1,2) 只用于三维数组
        q = q.reshape(batch_size,q_len,self.head_num,-1).transpose(0,2,1,3)   # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.reshape(batch_size,k_v_len,self.head_num,-1).transpose(0,2,1,3)
        v = v.reshape(batch_size,k_v_len,self.head_num,-1).transpose(0,2,1,3)

        k = k.transpose(0,1,3,2)

        atten_score = np.matmul(q,k)/np.sqrt(self.head_dim)   # [batch_size, num_heads, q_len, k_v_len]

        if mask is not None:
            mask_expanded = mask[:, np.newaxis, :, :]
            atten_score = np.where(mask_expanded,atten_score,-1e9)

        atten_weight = self.softmax(atten_score)
        atten_weight = self.Dropout(atten_weight)
        output = np.matmul(atten_weight,v) # [batch_size, num_heads, q_len, head_dim]
        output = output.transpose(0,2,1,3).reshape(batch_size,q_len,-1)

        return np.dot(output,self.out_proj)


if __name__ == '__main__' :
    head_num =8
    batch_size = 1
    emd_size = 128
    q_len = 5
    q_k_size = emd_size // 8
    v_size = emd_size // 8

    x = np.random.randn(batch_size,q_len,emd_size)
    self_atten = multi_attention(emd_size=emd_size,q_k_size=q_k_size,v_size=v_size,head_num = head_num)

    mask = np.random.randn(batch_size,q_len,q_len)
    mask = mask.astype(bool)
    print('多头的自注意力结果', self_atten.forward(x, x, mask).shape)

    batch_size = 1  # 批量也就是句子的数量
    emd_size = 128  # 一个token嵌入的维度
    q_seq_len = 5  # q源的token长度
    q_k_size = emd_size//8  # q和k的嵌入维度/head
    k_v_seq_len = 7  # k_v源的token长度
    v_size = emd_size//8  # v的嵌入维度/head
    head=8 # 头的数量

    x_q = np.random.rand(batch_size, q_seq_len, emd_size)
    x_k_v = np.random.rand(batch_size, k_v_seq_len, emd_size)

    cross_atten = multi_attention(emd_size=emd_size, q_k_size=q_k_size, v_size=v_size,head_num=head)
    mask = np.random.randn(batch_size, q_seq_len, k_v_seq_len)
    mask = mask.astype(bool)  #torch方法mask.bool()

    print('多头的交叉注意力结果', cross_atten.forward(x_q, x_k_v, mask).shape)