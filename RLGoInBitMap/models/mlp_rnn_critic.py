import torch.nn as nn
import torch
from torch.autograd import Variable


class Value(nn.Module): #类似判别器
    def __init__(self, state_dim, hidden_size=(128, 128),lstmNum = 3, activation='tanh'):
        super().__init__()
        '''
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.state_dim = state_dim
        self.hidden_size_rnn = state_dim
        self.num_layers = lstmNum


        self.lstm = nn.LSTM( input_size=self.state_dim, hidden_size=self.hidden_size_rnn,num_layers=self.num_layers,batch_first=True) #, batch_first=True))


        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)
        '''
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.num_layers = lstmNum

        self.lstm = nn.LSTM(input_size=state_dim, hidden_size=state_dim, num_layers=2,
                            batch_first=True)  # , batch_first=True))

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def initHidden(self,batch_size):
        if self.lstm.bidirectional:
            return (torch.zeros(self.num_layers*2,batch_size,self.hidden_size_rnn),torch.zeros(self.num_layers*2,batch_size,self.hidden_size_rnn))
        else:
            return (torch.zeros(self.num_layers ,batch_size,self.hidden_size_rnn),torch.zeros(self.num_layers,batch_size,self.hidden_size_rnn))
    '''
    def initHidden(self,batch_size):
        if self.lstm.bidirectional:
            return (Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size_rnn)),Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size_rnn)))
        else:
            return (Variable(torch.zeros(self.num_layers ,batch_size,self.hidden_size_rnn)),Variable(torch.zeros(self.num_layers,batch_size,self.hidden_size_rnn)))
'''

    def forward(self, input):
        '''
        self.hidden = self.initHidden(input.size(0))
        out, self.hidden = self.lstm(input, self.hidden)

        x = torch.transpose(out, 0, 1)
        #x = out[:,-1,:]
        #print(out.size())
'''
        #self.hidden =(torch.randn(1 ,1,33),torch.randn(1,1,33))
        out, _ = self.lstm(input)
        x = torch.transpose(out, 0, 1)
        x = x[-1]
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value