import numpy as np



#卷积层
class ConvolutionalLayer(object):
    '''
    N = (W-F+2P)/S + 1
    N-输出图片大小
    W-输入图片大小、
    F-卷积核大小
    P-填充
    S-步长
    
    类的成员:卷积核大小 填充值 步长 输入通道数 输出通道数
    对应kernel_size padding stride channel_in channel_out
    '''
    
    def __init__(self,kernel_size,padding,stride,channel_in,channel_out):
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.channel_in = channel_in
        self.channel_out = channel_out
    
    #初始化权重矩阵
    def init_param(self,std=0.01):
        '''
        输入通道数为channel_in 输出通道数为channel_out 
        channel_out个卷积核 每个卷积核深度为channel_in（因为要对channel_in个输入通道做扫描）
        每个卷积核一开始对channel_in个通道扫描 
        然后对channel_in个卷积后的结果做相加 得到channel_out个特征图
        
        channel_out个卷积核 每个卷积核大小为kernel_size*kernel_size 每个卷积核深度为channel_in
        '''  
        #loc为指定正态分布的均值  生成的随机数会围绕loc为中心
        #scale为正态分布的标准差
        #size为大小
        self.weight = np.random.normal(loc=0.0,scale=std,size=(self.channel_in,self.kernel_size,self.kernel_size,self.channel_out))
        self.bias = np.zeros([self.channel_out])
        
        def forward(self,input):
            self.input = input #[NCHW]
            #padding
            #注意是获取值shape 不是获取所有
            height = self.input.shape[2] + 2*self.padding
            width = self.input.shape[3] + 2*self.padding
            #创建padding之后的大矩阵
            padded = np.zeros(self.input.shape[0],self.input.shape[1],height,width)
            #padded height从0-self.input.shape[2] + 2*self.padding
            padded[:,:,self.padding:self.padding+self.input.shape[2],
                   self.padding:self.padding+self.input.shape[2]] = self.input
            
            #N = (W-F+2P)/S + 1 
            #使用//向下取整
            self.height_out = (self.input.shape[2] + 2*self.padding - self.kernel_size) // self.stride +1  
            self.width_out = (self.input.shape[3] + 2*self.padding - self.kernel_size) // self.stride +1
            