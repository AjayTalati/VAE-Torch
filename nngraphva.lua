-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
-- require 'xlua'

--Necessary for low variance estimator
require 'KLDCriterion'

--For loading data files
require 'load'

data = load32()

dim_input = data.train:size(2) 
dim_hidden = 10
HU_enc = 400
HU_dec = 401

batchSize = 100

torch.manualSeed(1)

input = nn.Identity()()
hidden_enc = nn.Linear(dim_input, HU_enc)(input)
act_enc = nn.Tanh()(hidden_enc)

log_std = nn.Linear(HU_enc, dim_hidden)(act_enc)
mean = nn.Linear(HU_enc, dim_hidden)(act_enc)
eps = nn.Identity()()

--KLD expects log(sigma)
std = nn.Exp()(log_std) -- only do this for sampling
sample_std = nn.CMulTable()({std, eps})

sample = nn.CAddTable()({mean, sample_std})

hidden_dec = nn.Linear(dim_hidden, HU_dec)(sample)
act_dec = nn.Tanh()(hidden_dec)

out_dec = nn.Linear(HU_dec, dim_input)(act_dec)
reconstruction = nn.Sigmoid()(out_dec)

va = nn.gModule({input, eps}, {reconstruction})
-- encoder = nn.gModule({input}, {mean, log_std})

x1 = torch.rand(dim_input)
x2 = torch.rand(dim_hidden)

--Uncomment to get structure of the Variational Autoencoder
-- graph.dot(va.fg, 'Variational Autoencoder', 'VA')

--Binary cross entropy term
BCE = nn.BCECriterion()
BCE.sizeAverage = false --I think I can turn this off
KLD = nn.KLDCriterion()

parameters, gradients = va:getParameters()

config = {
    learningRate = -0.03,
}

state = {}


epoch = 0
while true do
    epoch = epoch + 1
    local lowerbound = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train:size(1))

    --Make sure batches are always batchSize
    local N = data.train:size(1) - (data.train:size(1) % batchSize)
    local N_test = data.test:size(1) - (data.test:size(1) % batchSize)

    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, data.train:size(1))

        local batch = torch.Tensor(batchSize,data.train:size(2))

        local k = 1
        for j = i,i+batchSize-1 do
            batch[k] = data.train[shuffle[j]]:clone() 
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            va:zeroGradParameters()

            local f = va:forward(batch)
            local err = - BCE:forward(f, batch)
            local df_dw = BCE:backward(f, batch):mul(-1)

            va:backward(batch,df_dw)

            local KLDerr = KLD:forward(va:get(1).output, batch)
            local de_dw = KLD:backward(va:get(1).output, batch)

            encoder:backward(batch,de_dw)

            local lowerbound = err  + KLDerr

            return lowerbound, gradients
        end

        x, batchlowerbound = optim.adagrad(opfunc, parameters, config, state)
        lowerbound = lowerbound + batchlowerbound[1]
    end

    print("\nEpoch: " .. epoch .. " Lowerbound: " .. lowerbound/N .. " time: " .. sys.clock() - time)

    if lowerboundlist then
        lowerboundlist = torch.cat(lowerboundlist,torch.Tensor(1,1):fill(lowerbound/N),1)
    else
        lowerboundlist = torch.Tensor(1,1):fill(lowerbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/lowerbound.t7', torch.Tensor(lowerboundlist))
    end

end
