import torch


# 实现一个num_layers层的LSTM-RNN
class RNN(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super(RNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
								  batch_first=True)

	def forward(self, input):
		# input应该为(batch_size,seq_len,input_szie)
		self.hidden = self.initHidden(input.size(0))
		out, self.hidden = self.lstm(input, self.hidden)
		return out, self.hidden

	def initHidden(self, batch_size):
		if self.lstm.bidirectional:
			return (torch.rand(self.num_layers * 2, batch_size, self.hidden_size),
					torch.rand(self.num_layers * 2, batch_size, self.hidden_size))
		else:
			return (torch.rand(self.num_layers, batch_size, self.hidden_size),
					torch.rand(self.num_layers, batch_size, self.hidden_size))


input_size = 12
hidden_size = 10
num_layers = 3
batch_size = 2
model = RNN(input_size, hidden_size, num_layers)
# input (seq_len, batch, input_size) 包含特征的输入序列，如果设置了batch_first，则batch为第一维
input = torch.rand(2, 4, 12)
a,b = model(input)
print(a.size())