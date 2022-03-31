function [ps,pf]=MMO_CLRPSO(func_name,VRmin,VRmax,n_obj,Particle_Number,Max_Gen)
%% Initialize parameters
    n_var=size(VRmin,2);               %Obtain the dimensions of decision space
    Max_FES=Max_Gen*Particle_Number;   %Maximum fitness evaluations
    n_PBA=5;                           %Maximum size of PBA. The algorithm will perform better without the size limit of PBA. But it will time consuming.
    %n_NBA=10*n_PBA;                     %Maximum size of NBA   
    cc=[2.05 2.05];                    %Acceleration constants in PSO
    iwt=0.7298;                        %Inertia weight

%% Initialize particles' positions and velocities
    mv=0.5*(VRmax-VRmin);  %initialized velocity
    VRmin=repmat(VRmin,Particle_Number,1);
    VRmax=repmat(VRmax,Particle_Number,1);
    Vmin=repmat(-mv,Particle_Number,1);  %速度最小值边界
    Vmax=-Vmin;
    pos=VRmin+(VRmax-VRmin).*rand(Particle_Number,n_var); %initialize the positions of the particles
        
    %%Cluster Population
    subsize=10;  %子种群的个数
    num=Particle_Number/subsize;
    n_GBA=10*subsize;  %档案size
    n_NBA=3*n_GBA;  %
    pos=sortrows(pos,1); %按某一列排序
    particle_position=Cluster(pos,subsize,n_var);%返回值是由子种群组成的num*1的元包矩阵

    %calculate the objective value of particle,  initialize GBA, NBA
    for k=1:length(particle_position)
        for j=1:subsize
            particle_position{k,1}(j,n_var+1:n_var+n_obj)=feval(func_name,particle_position{k,1}(j,1:n_var));
            particle_position{k,j+1}=particle_position{k,1}(j,:);
            %——————————————————————————
            %元包的第k行的第一个元素是第k个子种群的个体position形成的矩阵，后面是每一个子种群的个体的pbest的矩阵，每个pbest可能有多个
            %形成一个num*（subsize+1）的元包矩阵
            %————————————————————————           
         end
        velocity{k,1}=Vmin(1:subsize,:)+2.*Vmax(1:subsize,:).*rand(subsize,n_var);
        temp_particle=non_domination_scd_sort(particle_position{k,1}, n_obj, n_var);
        tempindex=find(temp_particle(:,n_var+n_obj+1)==1);
        GBA{k,1}=temp_particle(tempindex,1:n_var+n_obj);
        subgbest{k,1}=GBA{k,1}(1,1:n_var+n_obj);
        for j=1:subsize
            if subgbest{k,1}==particle_position{k,1}(j,:)
                subgbest{k,2}=velocity{k,1}(j,:);
                break
            end
        end
        NBA{k,1}=GBA{k,1};
    end
    fitcount=Particle_Number; %统计FES        
%main loop
for i=1:Max_Gen
    %subpopulation evolution
    for k=1:num
        for m=1:subsize
            pbest_m=particle_position{k,m+1};
            pbest=pbest_m(1,:);
            
            %update velocity
            velocity{k,1}(m,:)=iwt.*velocity{k,1}(m,:)+cc(1).*rand(1,n_var).*(pbest(1,1:n_var)-particle_position{k,1}(m,1:n_var))...
                +cc(2).*rand(1,n_var).*(subgbest{k,1}(1,1:n_var)-particle_position{k,1}(m,1:n_var));
            velocity{k,1}(m,:)=(velocity{k,1}(m,:)>mv).*mv+(velocity{k,1}(m,:)<=mv).*velocity{k,1}(m,:);
            velocity{k,1}(m,:)=(velocity{k,1}(m,:)<(-mv)).*(-mv)+(velocity{k,1}(m,:)>=(-mv)).*velocity{k,1}(m,:);
            
            %update position
            particle_position{k,1}(m,1:n_var)=particle_position{k,1}(m,1:n_var)+velocity{k,1}(m,:);
            particle_position{k,1}(m,1:n_var)=((particle_position{k,1}(m,1:n_var)>=VRmin(1,:))&(particle_position{k,1}(m,1:n_var)<=VRmax(1,:))).*particle_position{k,1}(m,1:n_var)...
                +(particle_position{k,1}(m,1:n_var)<VRmin(1,:)).*(VRmin(1,:)+0.25.*(VRmax(1,:)-VRmin(1,:)).*rand(1,n_var))+(particle_position{k,1}(m,1:n_var)>VRmax(1,:)).*(VRmax(1,:)-0.25.*(VRmax(1,:)-VRmin(1,:)).*rand(1,n_var));
            
            %calculate objective space
            temp_fit=feval(func_name,particle_position{k,1}(m,1:n_var));
            particle_position{k,1}(m,n_var+1:n_var+n_obj)=temp_fit;
            fitcount=fitcount+1;
            
            %estimate the dominance relation of past and present pbest.
                        
            %Pbest is replaced if x dominates pbest, otherwise if x is mutually non-dominating with pbest,...
            %pbest has 50 probability to be replaced .
%             x_and_pbest=[particle_position{k,1}(m,1:n_var+n_obj);pbest(1,1:n_var+n_obj)];
%             temp_x_and_pbest=non_domination_scd_sort(x_and_pbest, n_obj, n_var);
%             tempindex1=find(temp_x_and_pbest(:,n_var+n_obj+1)==1);
%             if length(tempindex1)==2
%                 if rand<0.5
%                     pbest=particle_position{k,1}(m,1:n_var+n_obj);
%                 end
%             end
            
            %estimate pbest: change or unchange
            pbest_m=[particle_position{k,m+1};particle_position{k,1}(m,1:n_var+n_obj)];
            pbest_m=non_domination_scd_sort(pbest_m, n_obj, n_var);
            tempindex=find(pbest_m(:,n_var+n_obj+1)==1);
            pbest_m=pbest_m(tempindex,1:n_var+n_obj);
            if ismember(particle_position{k,m+1}(1,1:n_var+n_obj),pbest,'rows')==1
                particle_position{k,m+1}=[particle_position{k,m+1}(1,1:n_var+n_obj);pbest_m];%pbest unchange
                particle_position{k,m+1}=unique(particle_position{k,m+1},'rows');
            else
                particle_position{k,m+1}=pbest_m;%pbest change
            end
           
            [row_pbest_m,~]=size(pbest_m);
            if row_pbest_m>n_PBA
                pbest_m=pbest_m(1:n_PBA,1:n_var+n_obj);
                particle_position{k,m+1}=pbest_m;
            else
                particle_position{k,m+1}=pbest_m(:,1:n_var+n_obj);
            end
            
            %gbest更新与pbest合并比较
            gbest_m=[particle_position{k,m+1};subgbest{k,1}];
            gbest_m=non_domination_scd_sort(gbest_m, n_obj, n_var);
            tempindex=find(gbest_m(:,n_var+n_obj+1)==1);
            gbest_m=gbest_m(tempindex,1:n_var+n_obj);
            if ismember(subgbest{k,1},gbest_m,'rows')==0
                subgbest{k,1}=gbest_m(1,:);
                sungbest{k,2}=velocity{k,1}(m,:);
            end
            
            %gbest更新，与更新过的粒子合并比较（可以删除）
            gbest_m=[particle_position{k,1}(m,:);subgbest{k,1}];
            gbest_m=non_domination_scd_sort(gbest_m, n_obj, n_var);
            tempindex=find(gbest_m(:,n_var+n_obj+1)==1);
            gbest_m=gbest_m(tempindex,1:n_var+n_obj);
            if ismember(particle_position{k,1}(m,1:n_var+n_obj),gbest_m,'rows')==1
                subgbest{k,1}=particle_position{k,1}(m,:);
                subgbest{k,2}=velocity{k,1}(m,:);
            end
            
            %update GBA Archives. GBA is all non-dominated solution for
            %each subpopulation
            temp_GBA_k=[GBA{k,1};particle_position{k,1}(m,1:n_var+n_obj);particle_position{k,m+1}];
            temp_GBA_k=unique(temp_GBA_k,'rows');
            temp_GBA_k=non_domination_scd_sort(temp_GBA_k,n_obj,n_var);
            tempindex=find(temp_GBA_k(:,n_var+n_obj+1)==1);
            temp_GBA_k=temp_GBA_k(tempindex,1:n_var+n_obj);
            if length(tempindex)>n_GBA
                GBA{k,1}=temp_GBA_k(tempindex(1:n_GBA),1:n_var+n_obj);
            else
                GBA{k,1}=temp_GBA_k;
            end
        end
    end
    
    %update NBA:establish ring structure in NBA 
    for k=1:num
        if k==1
            tempNBA=GBA{k,1};
            tempNBA=[tempNBA;GBA{2,1}];
            tempNBA=[tempNBA;GBA{num,1}];
        elseif k==num
            tempNBA=GBA{k,1};
            tempNBA=[tempNBA;GBA{num-1,1}];
            tempNBA=[tempNBA;GBA{1,1}];
        else
            tempNBA=NBA{k,1};
            tempNBA=[tempNBA;GBA{k-1,1}];
            tempNBA=[tempNBA;GBA{k+1,1}];
        end
        tempNBA=unique(tempNBA,'rows');
        NBA_k=tempNBA;
        NBA_k=non_domination_scd_sort(NBA_k, n_obj, n_var);
        index_NBA_k=find(NBA_k(:,n_var+n_obj+1)==1);
        NBA_k=NBA_k(index_NBA_k,1:n_var+n_obj);
        [row_NBA_k,~]=size(NBA_k);
        if row_NBA_k>n_NBA
            NBA{k,1}=NBA_k(1:n_NBA,1:n_var+n_obj);   
        else
            NBA{k,1}=NBA_k(:,1:n_var+n_obj);
        end
    end
    
    %local search:employ local PSO
    for k=1:num
        pbest=GBA{k,1}(1,1:n_var+n_obj);
        
        %update velocity
        temp_vel_subgbest=iwt.*subgbest{k,2}+cc(1).*rand(1,n_var).*(pbest(1,1:n_var)-subgbest{k,1}(1,1:n_var))...
            +cc(2).*rand(1,n_var).*(NBA{k,1}(1,1:n_var)-subgbest{k,1}(1,1:n_var));
        temp_vel_subgbest=(temp_vel_subgbest>mv).*mv+(temp_vel_subgbest<=mv).*temp_vel_subgbest;
        temp_vel_subgbest=(temp_vel_subgbest<(-mv)).*(-mv)+(temp_vel_subgbest>=(-mv)).*temp_vel_subgbest;
        
        %update position
        temp_position=subgbest{k,1}(1,1:n_var)+subgbest{k,2};
        temp_position=((temp_position(1,:)>=VRmin(1,:))&(temp_position(1,:)<=VRmax(1,:))).*temp_position(1,:)...
            +(temp_position(1,:)<VRmin(1,:)).*(VRmin(1,:)+0.25.*(VRmax(1,:)-VRmin(1,:)).*rand(1,n_var))+(temp_position(1,:)>VRmax(1,:)).*(VRmax(1,:)-0.25.*(VRmax(1,:)-VRmin(1,:)).*rand(1,n_var));
        temp_position(:,n_var+1:n_var+n_obj)=feval(func_name,temp_position(1,1:n_var));
        
        %update the gbest of each subpopulation
        temp_subgbest=[temp_position;subgbest{k,1};GBA{k,1}];
        temp_subgbest=non_domination_scd_sort(temp_subgbest,n_obj,n_var);
        indextemp=find(temp_subgbest(:,n_var+n_obj+1)==1);
        temp_subgbest=temp_subgbest(indextemp,1:n_var+n_obj);
        if ismember(temp_position,temp_subgbest,'rows')==1
            GBA{k,1}=temp_subgbest;
            subgbest{k,1}=GBA{k,1}(1,:);
            subgbest{k,2}=temp_vel_subgbest;
        end
        
%         %if ismember(subgbest{k,1},temp_subgbest,'rows')==0
%             if ismember(temp_position,temp_subgbest,'rows')==1
%                 subgbest{k,1}=temp_position;
%                 subgbest{k,2}=temp_vel_subgbest;
%             end
%         %end
        
        %subgbest{k,1}=temp_position;
        fitcount=fitcount+1;
    end
   
    if fitcount>Max_FES
        break;
    end
end

%output non-dominated solution
tempEXA=cell2mat(GBA);
tempEXA=unique(tempEXA,'rows');
len1=length(tempEXA);
tempEXA=non_domination_scd_sort(tempEXA(:,1:n_var+n_obj), n_obj, n_var);
tempindex=find(tempEXA(:,n_var+n_obj+1)==1);% Find the index of the first rank particles
len2=length(tempindex);
ps=tempEXA(tempindex,1:n_var);
pf=tempEXA(tempindex,n_var+1:n_var+n_obj);

end



function f = non_domination_scd_sort(x, n_obj, n_var)
% non_domination_scd_sort:  sort the population according to non-dominated relationship and special crowding distance
%% Input：
%                      Dimension                      Description
%      x               num_particle x n_var+n_obj     population to be sorted     
%      n_obj           1 x 1                          dimensions of objective space
%      n_var           1 x 1                          dimensions of decision space

%% Output:
%              Dimension                                  Description
%      f       N_particle x (n_var+n_obj+4)               Sorted population  
%    in f      the (n_var+n_obj+1)_th column stores the front number
%              the (n_var+n_obj+2)_th column stores the special crowding distance   
%              the (n_var+n_obj+3)_th column stores the crowding distance in decision space
%              the (n_var+n_obj+4)_th column stores the crowding distance in objective space


    [N_particle, ~] = size(x);% Obtain the number of particles

% Initialize the front number to 1.
    front = 1;

% There is nothing to this assignment, used only to manipulate easily in
% MATLAB.
    F(front).f = [];
    individual = [];

%% Non-Dominated sort. 

    for i = 1 : N_particle
        % Number of individuals that dominate this individual
        individual(i).n = 0; 
        % Individuals which this individual dominate
        individual(i).p = [];
        for j = 1 : N_particle
            dom_less = 0;
            dom_equal = 0;
            dom_more = 0;
            for k = 1 : n_obj
                if (x(i,n_var + k) < x(j,n_var + k))
                    dom_less = dom_less + 1;
                elseif (x(i,n_var + k) == x(j,n_var + k))  
                    dom_equal = dom_equal + 1;
                else
                    dom_more = dom_more + 1;
                end
            end
            if dom_less == 0 && dom_equal ~= n_obj
                individual(i).n = individual(i).n + 1;
            elseif dom_more == 0 && dom_equal ~= n_obj
                individual(i).p = [individual(i).p j];
            end
        end   
        if individual(i).n == 0
            x(i,n_obj + n_var + 1) = 1;
            F(front).f = [F(front).f i];
        end
    end
% Find the subsequent fronts
    while ~isempty(F(front).f)
       Q = [];
       for i = 1 : length(F(front).f)
           if ~isempty(individual(F(front).f(i)).p)
                for j = 1 : length(individual(F(front).f(i)).p)
                    individual(individual(F(front).f(i)).p(j)).n = ...
                        individual(individual(F(front).f(i)).p(j)).n - 1;
                    if individual(individual(F(front).f(i)).p(j)).n == 0
                        x(individual(F(front).f(i)).p(j),n_obj + n_var + 1) = ...
                            front + 1;
                        Q = [Q individual(F(front).f(i)).p(j)];
                    end
               end
           end
       end
       front =  front + 1;
       F(front).f = Q;
    end
% Sort the population according to the front number
    [~,index_of_fronts] = sort(x(:,n_obj + n_var + 1));
    for i = 1 : length(index_of_fronts)
        sorted_based_on_front(i,:) = x(index_of_fronts(i),:);
    end
    current_index = 0;

%% SCD. Special Crowding Distance

    for front = 1 : (length(F) - 1)
  
        crowd_dist_obj = 0;
        y = [];
        previous_index = current_index + 1;
        for i = 1 : length(F(front).f)
            y(i,:) = sorted_based_on_front(current_index + i,:);%put the front_th rank into y
        end
        current_index = current_index + i;
   % Sort each individual based on the objective
        sorted_based_on_objective = [];
        for i = 1 : n_obj+n_var
            [sorted_based_on_objective, index_of_objectives] = ...
                sort(y(:,i));
            sorted_based_on_objective = [];
            for j = 1 : length(index_of_objectives)
                sorted_based_on_objective(j,:) = y(index_of_objectives(j),:);
            end
            f_max = ...
                sorted_based_on_objective(length(index_of_objectives), i);
            f_min = sorted_based_on_objective(1,  i);

            if length(index_of_objectives)==1
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;  %If there is only one point in current front
            elseif i>n_var
                % deal with boundary points in objective space
                % In minimization problem, set the largest distance to the low boundary points and the smallest distance to the up boundary points
                y(index_of_objectives(1),n_obj + n_var + 1 + i) = 1;
                y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)=0;
            else
                % deal with boundary points in decision space
                % twice the distance between the boundary points and its nearest neibohood 
                 y(index_of_objectives(length(index_of_objectives)),n_obj + n_var + 1 + i)...
                    = 2*(sorted_based_on_objective(length(index_of_objectives), i)-...
                sorted_based_on_objective(length(index_of_objectives) -1, i))/(f_max - f_min);
                 y(index_of_objectives(1),n_obj + n_var + 1 + i)=2*(sorted_based_on_objective(2, i)-...
                sorted_based_on_objective(1, i))/(f_max - f_min);
            end
             for j = 2 : length(index_of_objectives) - 1
                next_obj  = sorted_based_on_objective(j + 1, i);
                previous_obj  = sorted_based_on_objective(j - 1,i);
                if (f_max - f_min == 0)
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = 1;
                else
                    y(index_of_objectives(j),n_obj + n_var + 1 + i) = ...
                         (next_obj - previous_obj)/(f_max - f_min);
                end
             end
        end
    %% Calculate distance in decision space
        crowd_dist_var = [];
        crowd_dist_var(:,1) = zeros(length(F(front).f),1);
        for i = 1 : n_var
            crowd_dist_var(:,1) = crowd_dist_var(:,1) + y(:,n_obj + n_var + 1 + i);
        end
        crowd_dist_var=crowd_dist_var./n_var;
        avg_crowd_dist_var=mean(crowd_dist_var);
    %% Calculate distance in objective space
        crowd_dist_obj = [];
        crowd_dist_obj(:,1) = zeros(length(F(front).f),1);
        for i = 1 : n_obj
            crowd_dist_obj(:,1) = crowd_dist_obj(:,1) + y(:,n_obj + n_var + 1+n_var + i);
        end
        crowd_dist_obj=crowd_dist_obj./n_obj;
        avg_crowd_dist_obj=mean(crowd_dist_obj);
    %% Calculate special crowding distance
        special_crowd_dist=zeros(length(F(front).f),1);
        for i = 1 : length(F(front).f)
            if crowd_dist_obj(i)>avg_crowd_dist_obj||crowd_dist_var(i)>avg_crowd_dist_var
                special_crowd_dist(i)=max(crowd_dist_obj(i),crowd_dist_var(i)); % Eq. (6) in the paper
            else
                special_crowd_dist(i)=min(crowd_dist_obj(i),crowd_dist_var(i)); % Eq. (7) in the paper
            end
        end
        y(:,n_obj + n_var + 2) = special_crowd_dist;
        y(:,n_obj+n_var+3)=crowd_dist_var;
        y(:,n_obj+n_var+4)=crowd_dist_obj;
        [~,index_sorted_based_crowddist]=sort(special_crowd_dist,'descend');%sort the particles in the same front according to SCD
        y=y(index_sorted_based_crowddist,:);
        y = y(:,1 : n_obj + n_var+4 );
        z(previous_index:current_index,:) = y;
    end
    
f = z();
end
    
    
   
    
            
            
            
    
    