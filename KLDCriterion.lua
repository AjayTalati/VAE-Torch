local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(mean, log_sigma)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local log_sigma_sq = torch.mul(log_sigma, 2)
    local mean_sq = torch.pow(mean, 2)

    local KLDelement = log_sigma_sq:add(1):add(-1, mean_sq):add(-1, log_sigma_sq:exp())

    self.output = 0.5 * torch.sum(KLDelement)
    return self.output
end

function KLDCriterion:updateGradInput(mean, sigma)
	self.gradInput = {}
    self.gradInput[1] = (-mean):clone()
    -- self.gradInput[2] = (-torch.exp(input[2])):add(1):mul(0.5)

    --Not sure if - or :exp() takes precedence so I added extra brackets
    self.gradInput[2] = (-(torch.mul(input[2],2):exp())):add(1)

    return self.gradInput
end
