function C=Cluster(x,subsize,n_var)
num=1;
while(isempty(x)~=1)
    subpop=[];
    subpop(1,1:n_var)=x(1,1:n_var);
    x(1,:)=[];
    for k=2:subsize
        d=pdist2(subpop,x);
        d=mean(d);
        row=find(d==(min(d)));
        subpop=[subpop;x(row,1:n_var)];
        x(row,:)=[];
    end
    C{num,1}=subpop;
    num=num+1;
end

    
        