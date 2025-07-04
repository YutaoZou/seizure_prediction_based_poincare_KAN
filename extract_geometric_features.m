function [geome_features_all]=extract_geometric_features(subject)
% Extract geometric features
%
geome_features_all=[];
for i=1:length(subject)
    % Storage feature initialization
    geome_features_one_sample=zeros(1,270);

    for v=1:18

        std = zeros(1,5); % Store STD
        sdtc = zeros(1,5); % Store SDTC
        sshd = zeros(1,5); % Store SSHD

        % Discrete wavelet transform decomposes signals
        % Store the decomposed sub-signals
        signal_deconmposition= cell(5,1);
        num_sdec= length(signal_deconmposition); % Decompose the number
    
        % First-level decomposition
        [ca1,cd1]=dwt(subject{i}(v,:),'db6');
        signal_deconmposition{1}=cd1;
    
        % Two-level decomposition
        [ca2,cd2]=dwt(ca1,'db6');
        signal_deconmposition{2}=cd2;
    
        % Three-level decomposition
        [ca3,cd3]=dwt(ca2,'db6');
        signal_deconmposition{3}=cd3;
    
        % Four-level decomposition
        [ca4,cd4]=dwt(ca3,'db6');
        signal_deconmposition{4}=cd4;
        signal_deconmposition{5}=ca4;
    
        % Constructing the Poincare plot and extracting geometric features
        poincare_sequence = cell(num_sdec,1);    
        for u=1:num_sdec        
            length_sdec= length(signal_deconmposition{u}); % Decomposition length   
            
            y1_n=zeros(1,length_sdec-1);
            y2_n=zeros(1,length_sdec-1);

          % Poincare sequence
            for j=1:(length_sdec-1)
                 y1_n(j) = signal_deconmposition{u}(j);
                 y2_n(j) = signal_deconmposition{u}(j+1);                
            end

            poincare_sequence{u}=[y1_n;y2_n];

            % Standard descriptors of 2-D projection (STD)
            d1=(poincare_sequence{u}(1,:)-poincare_sequence{u}(2,:))./sqrt(2);
            SD1 = sqrt(var(d1));
            d2=(poincare_sequence{u}(1,:)+poincare_sequence{u}(2,:))./sqrt(2);
            SD2 = sqrt(var(d2));
            std(u) = pi*SD1*SD2;
            % Summation of distance from each point relative to a coordinate center (SDTC)
            dis_to_cen = sqrt(sum(poincare_sequence{u}.^2));
            sdtc(u) = sum(dis_to_cen);
            % Summation of the shortest distance from each point 
            % relative to the 45-degree line (SSHD) 
            sshd(u) = sum(abs(poincare_sequence{u}(1,:)-poincare_sequence{u}(2,:))./sqrt(2));           


        end

        geome_features_one_sample(1,(v-1)*15+1:v*15) = [std, sdtc ,sshd];

    end

    geome_features_all=[geome_features_all; geome_features_one_sample ];    
end

end