-- Based on JoinTable module

require 'nn'

local Reparametrize, parent = torch.class('nn.Reparametrize', 'nn.Module')

function Reparametrize:__init(dimension)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
end 

function Reparametrize:updateOutput(input)
    --Different eps for whole batch
    self.eps = torch.randn(input[2]:size(1),self.dimension)
    self.output = input[2]:exp():cmul(self.eps)

    -- Add the mean
    self.output:add(input[1])

    return self.output
end

function Reparametrize:updateGradInput(input, gradOutput)
    -- Derivative with respect to mean is 1
    self.gradInput[1] = gradOutput:clone()
    
    -- Derivative with respect to sigma
    self.gradInput[2] = input[2]:exp():cmul(self.eps)
    self.gradInput[2]:cmul(gradOutput)

    return self.gradInput
end
