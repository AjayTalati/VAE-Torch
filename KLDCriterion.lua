local KLDCriterion, parent = torch.class('nn.KLDCriterion', 'nn.Criterion')

function KLDCriterion:updateOutput(mean_sigma_eps)
    -- Appendix B from VAE paper: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    local log_sigma_sq = torch.mul(mean_sigma_eps[2], 2)
    local mean_sq = torch.pow(mean_sigma_eps[1], 2)

    local KLDelement = log_sigma_sq:add(1):add(-1, mean_sq):add(-1, torch.exp(log_sigma_sq))

    self.output = 0.5 * torch.sum(KLDelement)
    return self.output
end

function KLDCriterion:updateGradInput(mean_sigma_eps)
	self.gradInput = {}
    self.gradInput[1] = (-mean_sigma_eps[1]):clone()
    -- self.gradInput[2] = (-torch.exp(input[2])):add(1):mul(0.5)

    --Not sure if - or :exp() takes precedence so I added extra brackets
    self.gradInput[2] = (-(torch.mul(mean_sigma_eps[2],2):exp())):add(1)

    return self.gradInput
end
