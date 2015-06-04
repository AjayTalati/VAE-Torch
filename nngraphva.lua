-- Joost van Amersfoort - <joost@joo.st>
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
-- require 'xlua'

nngraph.setDebug(true)


--Necessary for low variance estimator
require 'KLDCriterion'

--For loading data files
require 'load'

local model_utils = require 'model_utils'

data = load32()

dim_input = data.train:size(2)
dim_hidden = 10
HU_enc = 400
HU_dec = 401

batch_size = 100

torch.manualSeed(1)

input = nn.Identity()()
hidden_enc = nn.Linear(dim_input, HU_enc)(input)
act_enc = nn.ReLU()(hidden_enc)

log_std_enc = nn.Linear(HU_enc, dim_hidden)(act_enc) 
mean_enc = nn.Linear(HU_enc, dim_hidden)(act_enc)

eps = nn.Identity()()
log_std_dec = nn.Identity()()
mean_dec = nn.Identity()()

std = nn.Exp()(log_std_dec)
sample_std = nn.CMulTable()({std, eps})

sample = nn.CAddTable()({mean_dec, sample_std})

hidden_dec = nn.Linear(dim_hidden, HU_dec)(sample)
act_dec = nn.Tanh()(hidden_dec)

out_dec = nn.Linear(HU_dec, dim_input)(act_dec)
reconstruction = nn.Sigmoid()(out_dec)

encoder = nn.gModule({input}, {mean_enc, log_std_enc})
-- decoder includes the reparametrization step
decoder = nn.gModule({mean_dec, log_std_dec, eps}, {reconstruction})

-- I want to do this, but not allowed due to dummy input module that is created
-- va = nn.gModule({input, eps}, {reconstruction})

--Maybe add encoder and decoder in a nn.Sequential, but not sure how to handle epsilon then

x1 = torch.rand(dim_input)
x2 = torch.rand(dim_hidden)

-- Uncomment to get structure of the Variational Autoencoder
-- graph.dot(va.fg, 'Variational Autoencoder', 'VA')

--Binary cross entropy term
BCE = nn.BCECriterion()
BCE.sizeAverage = false
KLD = nn.KLDCriterion()

parameters, gradients = model_utils.combine_all_parameters(encoder,decoder)

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

    --Make sure batches are always batch_size
    local N = data.train:size(1) - (data.train:size(1) % batch_size)
    local N_test = data.test:size(1) - (data.test:size(1) % batch_size)

    for i = 1, N, batch_size do
        xlua.progress(i+batch_size-1, data.train:size(1))

        local batch = torch.Tensor(batch_size,data.train:size(2))

        local k = 1
        for j = i,i+batch_size-1 do
            batch[k] = data.train[shuffle[j]]:clone() 
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end

            encoder:zeroGradParameters()
            decoder:zeroGradParameters()

            local mean_sigma_eps = encoder:forward(batch)
            table.insert(mean_sigma_eps, 3, torch.randn(batch_size, dim_hidden))
            local reconstruction = decoder:forward(mean_sigma_eps)

            -- check what happens when I remove both minus
            local err = - BCE:forward(reconstruction, batch)
            local df_dw = BCE:backward(reconstruction, batch):mul(-1)

            local df_ddec = decoder:backward(batch, df_dw)
            table.remove(df_ddec,3) -- remove eps gradient

            encoder:backward(batch, df_ddec)

            local KLDerr = KLD:forward(mean_sigma_eps)
            local dKLD_dw = KLD:backward(mean_sigma_eps)

            encoder:backward(batch,dKLD_dw)

            local lowerbound = err + KLDerr

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
