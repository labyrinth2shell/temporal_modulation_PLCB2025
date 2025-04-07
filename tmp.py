import numpy as np
import itertools
import copy
import pickle
import random
from matplotlib import pylab as pl

    
class myNN:
    
    # parameters for networks
    N=100
    d_path,c=0.1,7
    
    inv_tau,inv_tau_y=0.004,0.01  #inv_tau is timescale of connectivity   # inv_tau=0.01 in some previous cases

    def __init__(self,_flg,_inet,Nround=1000,Nin=1,Npat=5,Tstim=100,is_circle=True):
        self.BETA,self.BETA_y=5,20
        self.THRS_STOP,self.THRS_STOP1=0.9,0.5
         
        self.J=(2*np.random.randint(0,2,(self.N,self.N))-1)*(1/np.sqrt(self.N-1))
        self.J-=np.diag(np.diag(self.J))
        self.Jxy=(self.c/np.sqrt(self.N))*(2*np.random.randint(0,2,(self.N,self.N))-1)*np.where(np.random.rand(self.N,self.N)<self.d_path,1,0)
        self.inet=_inet
        np.random.seed(seed=_inet)
        
        if _flg=="simple":
            self.flg =_flg
            self.Npat=Npat
            self.Nin=Nin
            self.Tdyn=500
            self.ipt_trgt=self.gen_ipt_trgt()
            self.inv_tau=0.01
            self.Nround=Nround
            self.Tstim=Tstim
            self.initx,self.inity=0.05,0.05
            self.is_circle=is_circle
            self.is_testrcl=False
            
        elif _flg=="DMS" or _flg=="flx_DMS":
            self.flg =_flg
            self.Ntrgt=int(self.N/2)
            
            self.Nround=Nround
            if _flg=="DMS":
                self.Npat_intrial=4
                self.Tdyn =40  
                self.Tdyn_del=20
                self.Tdyn_test=40
                self.flg1="_"
            else:
                self.Npat_intrial=8
                self.Tdyn =40
                self.Tdyn_del_s=25
                self.Tdyn_del_l=50
                self.N_cxt=(int)(self.N/4)
                self.Tdyn_test=60
            self.initx,self.inity=0.05,0.05
            self.ipt_trgt=self.gen_ipt_trgt()
            

        else:
            print("invalid flg in gen_ipt_trgt!")
            
            
    def gen_ipt_trgt(self):
        if self.flg=="simple":
            ipt=2*np.random.randint(0,2,(self.N,self.Nin))-1  # input patterns
            trgt=2*np.random.randint(0,2,(self.N,self.Npat))-1
        elif self.flg=="DMS":
            cues=2*np.random.randint(0,2,(self.N,2))-1
            delay=np.zeros((self.N,1))
            
            trgt_out=2*np.random.randint(0,2,(self.Ntrgt,2))-1
            rnd_out=2*np.random.randint(0,2,(self.N-self.Ntrgt,8))-1

            ipt,trgt=[],[]
            correct_trgt_id=[0,1,1,0]
            for i in range(4):
                tmp=[cues[:,i%2].reshape(-1,1),\
                     delay[:,0].reshape(-1,1),\
                     cues[:,i//2].reshape(-1,1)  ]
                ipt.append(copy.deepcopy(tmp))

                tmp=[ [], [], np.hstack((trgt_out[:,correct_trgt_id[i]],rnd_out[:,i])).reshape(-1,1),trgt_out[:,(correct_trgt_id[i]+1)%2]  ]
                trgt.append(copy.deepcopy(tmp))
                
        elif self.flg=="flx_DMS":
            cues=2*np.random.randint(0,2,(self.N-self.N_cxt,2))-1
            cxt_cues=2*np.random.randint(0,2,(self.N_cxt,2))-1
            delay=np.zeros((self.N-self.N_cxt,1))
            
            trgt_out=2*np.random.randint(0,2,(self.Ntrgt,2))-1
            rnd_out=2*np.random.randint(0,2,(self.N-self.Ntrgt,8))-1
            
            ipt,trgt=[],[]
            correct_trgt_id=[0,1,1,0]
            for j in range(2):
                for i in range(4):
                    tmp=[np.hstack((cues[:,i%2],cxt_cues[:,j])).reshape(-1,1),\
                         np.hstack((delay[:,0],cxt_cues[:,j])).reshape(-1,1),\
                         np.hstack((cues[:,i//2],cxt_cues[:,j])).reshape(-1,1)  ]
                    ipt.append(copy.deepcopy(tmp))
                    
                    tmp=[ [], [], np.hstack((trgt_out[:,correct_trgt_id[i]],rnd_out[:,i])).reshape(-1,1),trgt_out[:,(correct_trgt_id[i]+1)%2] ]
                    trgt.append(copy.deepcopy(tmp))
                    
        return [ipt,trgt]

    
    def init_every_set(self,is_rcl,idset,x,y):
        if self.flg=="simple":
            len_tmp=len(self.map_schedule)
            id_i,id_t=self.map_schedule[idset%len_tmp]
            ipt =self.ipt_trgt[0][:,id_i : id_i+1]
            trgt=self.ipt_trgt[1][:,id_t : id_t+1] 
            
            if not is_rcl or self.is_testrcl:
                ipt=ipt+0.01*np.random.randn(100,1)
                
            if self.is_circle:
                if idset>0 and not is_rcl:  # perturbation at switching a target to another during learning
                    x*=np.random.rand(self.N,1)
            else:
                if idset>0 and idset%len_tmp==0:
                    x=self.initx*(2*np.random.rand(self.N,1)-1) 
                    y=self.inity*(2*np.random.rand(self.N,1)-1)
                    
            return x,y,ipt,trgt,"_"
        

        else:
            if idset%3==0:  # set initial state at the beginning of trial
                x=self.initx*(2*np.random.rand(self.N,1)-1)  # during learning,  0.01
                y=self.inity*(2*np.random.rand(self.N,1)-1)
            else:  # when the input is switched, a little perturbation is applied
                x*=(1+0.1*(2*np.random.rand(self.N,1)-1))  # during learning,  0.01
                y*=(1+0.1*(2*np.random.rand(self.N,1)-1))

                
            nset=self.gset[idset//3]
            if self.flg=="DMS":
                idtmp=idset%3
                if idtmp==2 and self.flg1=="perturb_2ndstim" :
                    ipt=np.copy(self.ipt_trgt[0][nset][idtmp])
                    d=int(0.03*self.N)
                    pert_id=np.array(random.sample(list(np.arange(self.N)),d))
                    ipt[pert_id,:]*=-1
                else:
                    ipt=np.copy(self.ipt_trgt[0][nset][idtmp])
                trgt=np.copy(self.ipt_trgt[1][nset][idtmp]) # includes non-target pattern (zeros)
                nontrgt=np.copy(self.ipt_trgt[1][nset][3])

            elif self.flg=="flx_DMS":
                idtmp=idset%3
                ipt    =self.ipt_trgt[0][nset][idtmp]
                trgt   =self.ipt_trgt[1][nset][idtmp] # includes non-target pattern (zeros)
                nontrgt=self.ipt_trgt[1][nset][3]
            else:
                print("ivalid flg")
                
            return x,y,ipt,trgt,nontrgt

        
    
    def gen_Tset_gset(self,is_rcl):
        
        if self.flg=="DMS":
            tmp=[[self.Tdyn,self.Tdyn_del,self.Tdyn_test] for i in range(self.Npat_intrial)]
                        
            if not is_rcl:
                preTset=np.array([tmp for i in range(self.Nround)])
                pregset=np.array([np.random.permutation([0,1,2,3]) for i in range(self.Nround)])
            else:
                preTset=np.array([tmp for i in range(self.Nround)])
                pregset=np.array([[0,1,2,3] for i in range(self.Nround)])

            """
            if not is_rcl:
                nround=1000  # 500->1000 2023.9.1
                Nset=4*nround
                preTset=np.array([[self.Tdyn,self.Tdyn_del+np.random.uniform(-10,10),self.Tdyn_test] for i in range(Nset)])
                pregset=np.array([np.random.permutation([0,2,8,10]) for i in range(nround)])
                
            else:
                preTset=np.array([[self.Tdyn,self.Tdyn_del+np.random.uniform(-10,10),self.Tdyn_test] for i in range(4)])
                pregset=np.array([0,2,8,10])
            """
        else:
            tmp=[ [self.Tdyn,self.Tdyn_del_s,self.Tdyn_test] for i in range(4) ] + \
                [ [self.Tdyn,self.Tdyn_del_l,self.Tdyn_test] for i in range(4) ]
            
            if not is_rcl:
                preTset=np.array([tmp for i in range(self.Nround)])
                pregset=np.array([np.random.permutation([0,1,2,3,4,5,6,7]) for i in range(self.Nround)])
            else:
                preTset=np.array(tmp)
                pregset=np.array([0,1,2,3,4,5,6,7])

        return list(preTset.flatten()),list(pregset.flatten())
            
        
    
    ############################################################################################
    ############################################################################################
    ############################################################################################
    #x,y,dyn,dyn1,cnect,t_tic=dyf.calc_dyn(cnect,_vars,paras,ipt_trgt,funcs,is_rcl,0)
    def calc_dyn(self,_vars,is_rcl,is_save):
        dt=0.05
        entire_t=0
        x0,y0=_vars[0],_vars[1]
    
        nitr_rcd=1
        dyn,dyn1=[],[]
       
        if not is_rcl:
            cnt_lrn      =0   # counter after learning performacne reaches a criteria.

        if self.flg=="simple":
            if is_rcl:
                self.Tset    =self.Tdyn*np.ones(1)
            else:
                self.Tset    =self.Tdyn*np.ones(len(self.map_schedule)*self.Nround)
                
            if type(x0)!=np.ndarray:
                x0=self.initx*(2*np.random.rand(self.N,1)-1)
            if type(y0)!=np.ndarray:
                y0=self.inity*(2*np.random.rand(self.N,1)-1)
            x,y=x0.copy(),y0.copy()
                
        elif self.flg in ["DMS","flx_DMS"]:
            t_tic=[0]
            x,y  =self.initx*(2*np.random.rand(self.N,1)-1),self.inity*(2*np.random.rand(self.N,1)-1)
            self.Tset,self.gset=self.gen_Tset_gset(is_rcl)
            self.score_all=[]
             
        for idset in range(len(self.Tset)):
            x,y,ipt,trgt,nontrgt=self.init_every_set(is_rcl,idset,x,y)
                
            ##########    dynamics in a single section (1 input is applied)   ##########
            for it in np.arange(0,int(self.Tset[idset]/dt)):
                if self.flg=="DMS":
                    if is_rcl and idset%3==1 and it*dt>30:
                        self.BETA=self.BETArcl2
                    elif is_rcl and idset%3==2:
                        self.BETA=self.BETArcl2
                    elif is_rcl and idset%3==0:
                        self.BETA=self.BETArcl1
                if self.flg=="simple" and is_rcl and self.Nin!=1:
                    if it*dt>self.Tstim:
                        ipt =self.ipt_trgt[0][:,1 : 2]
                
                if is_rcl:
                    self.calc_next(x,y,ipt,dt)
                else:
                    if self.is_train(idset):
                        self.calc_next_lrn(x,y,ipt,trgt,dt)
                    else:
                        self.calc_next(x,y,ipt,dt)

                if not is_rcl and self.stop(idset,x,y,trgt,nontrgt,it*dt):
                    break

                if (self.flg=="simple" or (self.flg in ["DMS","flx_DMS"] and is_rcl) ) and it%nitr_rcd==0:
                    dyn.append(np.vstack((np.array([(entire_t+it)*dt]),np.copy(x))))
                    dyn1.append(np.vstack((np.array([(entire_t+it)*dt]),np.copy(y))))
            ##################      dyn  end    #############################
            
            entire_t+=it
            
            if self.flg=="simple":
                dyn.append(np.vstack((np.array([entire_t*dt]),np.copy(x))))
                dyn1.append(np.vstack((np.array([entire_t*dt]),np.copy(y))))

                if not is_rcl:
                    if idset%(len(self.map_schedule)*3)==len(self.map_schedule)*3-1:
                        print(idset)
                        self.x_final=x;  self.y_final=y
                        
                        tmp=self.check_score()
                        if tmp:
                            cnt_lrn+=1
                        else:
                            cnt_lrn=0
                           
                        if cnt_lrn==4:
                            break
                                
                
            elif self.flg  in ["DMS","flx_DMS"]:
                t_tic.append(entire_t*dt)

                if is_rcl:
                    dyn.append(np.vstack((np.array([entire_t*dt]),np.copy(x))))
                    dyn1.append(np.vstack((np.array([entire_t*dt]),np.copy(y))))
                else:
                    if idset%240==239:
                        score=self.check_score()
                        print(int(idset/(3*self.Npat_intrial)),np.mean(score))
                        self.score_all.append(score)
                        if np.mean(score)>0.8:
                            break
                        
                

                        
        if self.flg=="simple" or (self.flg in ["DMS","flx_DMS"] and is_rcl):
            dyn,dyn1=np.array(dyn),np.array(dyn1)
            
        
        if not is_rcl and is_save:
            if self.flg=="simple":
                #np.savez("temp_mod_results_inet%d_Npat%d_%g" % (self.inet,self.Npat,self.BETA),\
                #         [self.J,self.Jxy,self.ipt_trgt,y,dyn,dyn1])
                self.x_final=x
                self.y_final=y
                self.dyn_fast=dyn
                self.dyn_slow=dyn1

                if self.Nin!=1:
                    fname="mod_results_"+self.flg+"_inet%d_Npat%d_Nin%d_%g" % (self.inet,self.Npat,self.Nin,self.BETA)
                else:
                    #fname="mod_results_"+self.flg+"_inet%d_Npat%d_%g" % (self.inet,self.Npat,self.BETA)
                    fname="./used_data/results_"+self.flg+"_pert_inet%d_Npat%d_%g" % (self.inet,self.Npat,self.BETA)  # perturbed input is applied during learning
                f = open(fname,"wb")
                pickle.dump(self,f)
                f.close
            else:
                fname="./used_data/results_"+self.flg+"_inet%d_%g_%g" % (self.inet,self.BETA,self.gamma)
                f = open(fname,"wb")
                pickle.dump(self,f)
                f.close
            """
            else:
                #np.savez("temp_mod_results_DMS_inet%d_%g" % (self.inet,self.BETA),\
                #[self.J,self.Jxy,self.ipt_trgt,t_tic,score_all])
                
                #np.savez("temp_mod_results_DMS_flxdel_inet%d_%g_%g" % (self.inet,self.BETA,self.gamma),\
                #[self.J,self.Jxy,self.ipt_trgt,t_tic,score_all])  2023.8.30
                
                #np.savez("pert_mod_results_DMS_flxdel_inet%d_%g_%g" % (self.inet,self.BETA,self.gamma),\
                #         [self.J,self.Jxy,self.ipt_trgt,t_tic,self.score_all]) 
            """
        if self.flg=="simple":
            return x,y,dyn,dyn1
        else:
            if not is_rcl:
                return 
            else:
                return x,y,dyn,dyn1,t_tic
         
        
    ############################################################################################
    ############################################################################################
    ############################################################################################        



    
    def cond(self,i):
        if i==0 or i==3:
            return 0
        else:
            return 1
        
    def check_score(self):
        if self.flg in ["DMS","flx_DMS"]:
            _thrs=self.THRS_STOP
            _Ninit=20
            _score=np.zeros((_Ninit,self.Npat_intrial))

            NNtmp=copy.deepcopy(self)
            NNtmp.Nround=1
            NNtmp.BETArcl1=NNtmp.BETA
            NNtmp.BETArcl2=NNtmp.BETA
                
            for iinit in range(_Ninit):
                _x,_y,_rcl,_rcl1,_t_tic_rcl=NNtmp.calc_dyn(["_","_"],True,False)
                if self.flg in ["simple","DMS"]:
                    tmptrgt=np.hstack((self.ipt_trgt[1][0][2][:self.Ntrgt,:],self.ipt_trgt[1][1][2][:self.Ntrgt,:]))
                    t,dyn=_rcl[:,0,0],_rcl[:,1:self.Ntrgt+1,0]@tmptrgt/self.Ntrgt
                else:
                    tmptrgt=np.hstack((self.ipt_trgt[1][0][2][:self.Ntrgt,:],self.ipt_trgt[1][1][2][:self.Ntrgt,:]))
                    t,dyn=_rcl[:,0,0],_rcl[:,1:self.Ntrgt+1,0]@tmptrgt/self.Ntrgt

                for i in range(self.Npat_intrial):
                    t_strt,t_end=np.where(t==_t_tic_rcl[(i+1)*3-1])[0][0], np.where(t==_t_tic_rcl[(i+1)*3])[0][0]
                    for k in dyn[t_strt:t_end+1]:
                        if any(k>_thrs):
                            _score[iinit,i]=k[self.cond(i)]
                            break
            del NNtmp
            tmp=np.where(_score>_thrs,1,0)

            return tmp
        elif self.flg == "simple":
            NNtmp=copy.deepcopy(self)
            NNtmp.Nround=1
            NNtmp.Tdyn=3000
            x0=self.x_final
            y0=self.y_final
            NNtmp.is_testrcl=True
            
            _,_,dyntmp,_=NNtmp.calc_dyn([x0,y0],True,False)
            
            is_seq=NNtmp.check_iterations(dyntmp,nitr=4)
            if isinstance(is_seq[0], float):
                is_seq=True
            else:
                is_seq=False
                
            del NNtmp
            return is_seq
            
                    

            

    def stop(self,idset,x,y,trgt,nontrgt,t):
        if self.flg=="simple":
            if np.mean(x*trgt)>self.THRS_STOP and np.mean(x*y)>self.THRS_STOP1:    
                return True
        elif self.flg=="DMS":
            if idset%3==2 and t>2:
                if np.mean(x[:self.Ntrgt]*trgt[:self.Ntrgt])>self.THRS_STOP or np.mean(x[:self.Ntrgt]*nontrgt[:self.Ntrgt])>self.THRS_STOP:
                    return True
        else:
            if idset%3 == 2 and t>2:
                if np.mean(x[:self.Ntrgt]*trgt[:self.Ntrgt])>self.THRS_STOP or np.mean(x[:self.Ntrgt]*nontrgt[:self.Ntrgt])>self.THRS_STOP:
                    return True
            
        return False


    def is_train(self,idset):
        if self.flg=="simple":
            return True
        elif self.flg in ["DMS","flx_DMS"]:  # 最初の__init__でsimple, DMS以外は排除されているはずだが、もし今後__init__を変更したときのために、ここでも一応場合わけをしておく
            if idset%3==2:
                return True
            else:
                return False
        else:
            print("invalid flg in stop()")
        
        
    def check_iterations(self,dyntmp,nitr=2,thrs=90):
        tmp1=[]
        for i in range(self.Npat):
            tmptmp=self.detect_t_trans(dyntmp, i,thrs)
            if tmptmp==False:
                continue
            for k in tmptmp:
                tmp1+= [[i,k]]
        if len(tmp1)==0:
            return False,False
            

        Ttmp=np.array(tmp1)
        Ttmp=Ttmp[ np.argsort(Ttmp[:,1])  ]  # sorted by time
        # check dupulications
        #tmp=[i+1 for i,k in enumerate(zip(Ttmp[:-1],Ttmp[1:])) if int(k[0][0])==int(k[1][0])]
        #Ttmp=np.delete(Ttmp,tmp,0)

        tmplate=np.tile(np.arange(self.Npat),nitr).astype(int)
        for i in range(len(Ttmp)-nitr*self.Npat+1):
            if np.all(Ttmp[i:i+nitr*self.Npat,0].astype(int)==tmplate):
                return (Ttmp[i+nitr*self.Npat-1,1]-Ttmp[i,1])/nitr,Ttmp
        return False,False
       
            
    def detect_t_trans(self,dyn,ipat,thrs=80):
        prj_dyn=dyn[:,1:,0]@self.ipt_trgt[1]
        t=dyn[:,0,0]
        tmp=[it for it, v in enumerate(prj_dyn[:,ipat]) if v >thrs]
        if len(tmp)!=0:
            tmptmp=[list(g) for _, g in itertools.groupby(tmp, key=lambda n, c=itertools.count(): n - next(c))]
            t_trans=[t[k[0]] for k in tmptmp]
        else:
            t_trans=False

        return t_trans
    
    def detect_t_trans_1(self,dyn,ipat,thrs=80):  # detect transition times at ascending and descending phase
        prj_dyn=dyn[:,1:,0]@self.ipt_trgt[1]
        t=dyn[:,0,0]
        tmp=[it for it, v in enumerate(prj_dyn[:,ipat]) if v >thrs]
        if len(tmp)!=0:
            tmptmp=[list(g) for _, g in itertools.groupby(tmp, key=lambda n, c=itertools.count(): n - next(c))]
            t_trans=[[t[k[0]] for k in tmptmp],[t[k[-1]] for k in tmptmp]]
        else:
            t_trans=False

        return t_trans
    
    
    def calc_next_lrn(self,x,y,ipt,trgt,dt):
        f,fy,g=self.func_lrn(x,y,ipt,trgt)
        x+=f*dt
        y+=fy*dt*self.inv_tau_y
        self.J+=g*dt*self.inv_tau
        return

    def func_lrn(self,x,y,ipt,trgt):
        f,fy,m=self.func(x,y,ipt)
        if self.flg=="simple":
            err=trgt-x
        else:
            err=np.zeros(x.shape)
            err[:self.Ntrgt]=trgt[:self.Ntrgt]-x[:self.Ntrgt]
        g=(np.dot(err,x.T)-np.dot(np.diag((err*m).reshape(self.N)),self.J))/float(self.N)
        g-=np.diag(np.diag(g))
        return f,fy,g

    def calc_next(self,x,y,ipt,dt):
        f,fy,m=self.func(x,y,ipt)
        x+=f*dt
        y+=fy*dt*self.inv_tau_y

        return
    
    def func(self,x,y,ipt):
        m=np.dot(self.J,x)
        r=np.dot(self.Jxy,y)
        #f=np.tanh(self.BETA*(m+self.gamma_y*np.tanh(r)+self.gamma*ipt))-x
        f=np.tanh(self.BETA*(m+self.gamma_y*r+self.gamma*ipt))-x
        fy=np.tanh(self.BETA_y*x)-y
        return f,fy,m

    
    
######################################################################    
#####################   slow and fast neurons are within a network   ###############    
class including_model:
    # parameters for networks
    Nfast,Nslow=100,50
    N=Nfast+Nslow
    d_path,c=0.1,7
    BETA,BETA_y=2,20
    
    # parameters for dynamics
    gamma   =1.
    inv_tau,inv_tau_cell=0.01,np.ones((N,1))
    inv_tau_cell[Nfast:]=0.01
    inv_tau_cell_mat=np.sqrt(np.dot(inv_tau_cell,inv_tau_cell.T))
    
    # parameters for learning
    THRS_STOP=0.75
    
    
    def __init__(self,_inet):
        self.J=(2*np.random.randint(0,2,(self.N,self.N))-1)*(1/np.sqrt(self.N-1))
        self.J-=np.diag(np.diag(self.J))
        self.inet=_inet
        np.random.seed(seed=_inet)
        
        self.Npat=3
        self.Tdyn=200
        self.ipt_trgt=self.gen_ipt_trgt()

    def gen_ipt_trgt(self):
        ipt=2*np.random.randint(0,2,(self.N,1))-1  # input patterns
        trgt=2*np.random.randint(0,2,(self.N,self.Npat))-1
            
        return [ipt,trgt]
       
        
    def calc_dyn(self,_vars,is_rcl):
        dt=0.05
        entire_t=0
        x0=_vars[0]
    
        nitr_rcd=5
        dyn=[]
            
        if is_rcl:
            self.Tset    =self.Npat*self.Tdyn*np.ones(1)
            x=x0.copy()
        else:
            self.Tset    =self.Tdyn*np.ones(self.Npat*20)
            x=2*np.random.rand(self.N,1)-1
            

        for idset in range(len(self.Tset)):
            x,ipt,trgt=self.init_every_set(is_rcl,idset,x)

            for it in np.arange(0,int(self.Tset[idset]/dt)):
                if is_rcl:
                    self.calc_next(x,ipt,dt)
                else:
                    if self.is_train(idset):
                        self.calc_next_lrn(x,ipt,trgt,dt)
                    else:
                        self.calc_next(x,ipt,dt)

                if not is_rcl and self.stop(idset,x,trgt):
                    break

                if it%nitr_rcd==0:
                    dyn.append(np.vstack((np.array([(entire_t+it)*dt]),np.copy(x))))

            entire_t+=it
            dyn.append(np.vstack((np.array([entire_t*dt]),np.copy(x))))
                        
        dyn=np.array(dyn)
        if not is_rcl:
            np.savez("temp_mod_results_includingmodel_inet%d_Npat%d_%g" % (self.inet,self.Npat,self.BETA),[self.J,self.ipt_trgt,dyn])

        return x,dyn
  
    
    def init_every_set(self,is_rcl,idset,x):
        id_i=0
        id_t=idset%self.Npat
        ipt =self.ipt_trgt[0][:,id_i : id_i+1]
        trgt=self.ipt_trgt[1][:,id_t : id_t+1]
        
        if idset>0 and not is_rcl:
                x*=(1-np.sqrt(self.inv_tau_cell)*np.random.rand(self.N,1))
                
        return x,ipt,trgt

    def calc_next_lrn(self,x,ipt,trgt,dt):
        m=np.dot(self.J,x)
        f=np.tanh(self.BETA*(m + self.gamma*ipt)) - x
        err=(trgt-x)
        g=(np.dot(err,x.T)-np.dot(np.diag((err*m).reshape(self.N)),self.J))/float(self.N)
        g-=np.diag(np.diag(g))
        g*=self.inv_tau_cell_mat
        
        x+=self.inv_tau_cell*f*dt
        self.J+=g*dt*self.inv_tau
        return
    
    def calc_next(self,x,ipt,dt):
        m=np.dot(self.J,x)
        f=np.tanh(self.BETA*(m + self.gamma*ipt)) - x

        x+=self.inv_tau_cell*f*dt
        return
        
        
    def stop(self,idset,x,trgt):
        if np.mean(x*trgt)>self.THRS_STOP:    
            return True
        return False


    def is_train(self,idset):
        return True
        
        
