function [inputs,outputs] = scaleSOFC(inputs,outputs)

inputs.RatioHC = inputs.RatioHC/1e16;
inputs.RatioOC = inputs.RatioOC/1e16;
inputs.RatioSteamC = inputs.RatioSteamC/1e16;
inputs.Temperature = inputs.Temperature/1e18;
%
outputs.CO = outputs.CO/1e16;
outputs.H2 = outputs.H2/1e16;

end

