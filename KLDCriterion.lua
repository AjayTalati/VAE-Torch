local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(input, target)
    -- 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    -- local KLDelement = (input[2] + 1):add(-1,torch.pow(input[1],2)):add(-1,torch.exp(input[2]))
    local log_sigma_squared = torch.mul(input[2],2)
    local KLDelement = (log_sigma_squared + 1):add(-1,torch.pow(input[1],2)):add(-1,torch.exp(log_sigma_squared))
    self.output = 0.5 * torch.sum(KLDelement)
    return self.output
end

function KLDCriterion:updateGradInput(input, target)
	self.gradInput = {}
    self.gradInput[1] = (-input[1]):clone()
    self.gradInput[2] = (-torch.exp(input[2])):add(1)

    return self.gradInput
end
