classdef ds

    properties
        input
        output
    end
    
    methods
        function obj = ds(input,output)

            [~,iNt] = size(input);
            [~,oNt] = size(output);
            
            % validation 
            if length(unique([ iNt oNt])) ~= 1
                error('The row of input and output must be equal to length of tspan')
            end
            obj.input = input;
            obj.output = output;

        end
        
        function [inputs,outputs] = narx(obj,no)
            
            
            inputs = obj.input(:,(no+1):end);
            for i = 1:no
                inputs = [inputs; obj.output(:,(no-i+1):end-i) ];
            end
            outputs = obj.output(:,(no+1):end);

        end
        
        function [inputs,outputs] = serveral_narx(objs,no)
            inputs = [];
            outputs = [];
            for iobj = objs
                [in,out] = narx(iobj,no);
                inputs = [inputs in];
                outputs = [outputs out];
            end
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        function testplot(obj,net,no)
            subplot(4,1,1)
            plot(obj.input')
            title('input')
            legend('Location','bestoutside')
            subplot(4,1,2)
            plot(obj.output')
            ylim([-1 1])
            title('output real')
            legend('Location','bestoutside')
            subplot(4,1,3)
            
            [inputs_narx,output_narx] = narx(obj,no);

            SimOut = sim(net,inputs_narx);

            plot(SimOut')
            ylim([-1 1])
            title('output pred')
            legend('Location','bestoutside')

            subplot(4,1,4)
            
            plot(abs(SimOut'-output_narx'))
            ylim([0 0.5])
            title('Error')
            legend('Location','bestoutside')

        end
    end
end

