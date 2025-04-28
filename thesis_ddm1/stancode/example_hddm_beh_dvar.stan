// DDM for gaze task behavioral data only (with trial-level drift variability)
// 
// Requires Stan 2.35 or higher for 7-par wiener likelihood: wiener(alpha, tau, beta, delta, var_delta, var_beta, var_tau)
// https://mc-stan.org/docs/functions-reference/positive_lower-bounded_distributions.html#wiener-first-passage-time-distribution 

// This program defines a hierarchical drift diffusion model (Ratcliff 1978, 
// Ratcliff, 2008), with a non-centered parameterization. YES responses are 
// modeled as upper boundary responses and NO responses are modeled as
// lower boundary responses. 

data {
  int N_obs;                  // number of observations                          
  int N_subj;                 // number of subjects
  int N_cond;                 // number of gaze conditions (determines accuracy of responses)
  int N_cond_other;           // number of additional head*emo task conditions (don't determine accuracy of responses responses)
  int N_choice;               // number of choice alternatives 
  int N_groups;               // number of groups 
  array[N_obs] real RT;       // RT in seconds for each trial 
  array[N_obs] int subj;      // subj id's for each trial 
  array[N_obs] int choice;    // response for each trial (1=yes,2=no)
  array[N_obs] int cond;      // gaze cond for each trial (1=direct, 2=indirect) (determines accuracy of responses)
  array[N_obs] int cond_other;// cond for other head*emo conditions for each trial (1=forward head, neutral emo 2=forward head, fearful emo; 3=deviated head, neutral emo, 4=deviated head, fearful emo); (doesn't determine accuracy of responses)
  array[N_obs] int group;     // group id for each trial
  array[N_subj] real minRT;   // minimum RT in seconds for each subject 
  int fix_ndt;                //fix  NDT @ 0.2? (for par recovery); 1=yes, 0=no.
}

transformed data {
  array[N_subj] int subj_group;   // gives vector of group id's at subject-level 
  for (i in 1:N_obs){
    subj_group[subj[i]]=group[i];
  }
}

parameters { 
  
  // GROUP-level parameters (untransformed for non-centered parameterization)
  vector[N_groups] mu_grp_alpha_pr;           // threshold sep. group mean
  vector[N_groups] mu_grp_beta_pr;            // start point group mean
  vector[N_groups] mu_grp_delta_present_pr;   // drift rate group mean, stim present
  vector[N_groups] mu_grp_delta_absent_pr;    // drift rate group mean, stim absent
  vector[N_groups] mu_grp_ndt_pr;             // non-decision time group mean
  vector<lower=0>[N_groups] sig_grp_alpha_pr; // threshold sep. group SD
  vector<lower=0>[N_groups] sig_grp_beta_pr;  // start point group SD
  vector<lower=0>[N_groups] sig_grp_delta_pr; // drift rate group SD
  vector<lower=0>[N_groups] sig_grp_ndt_pr;   // non-decision time group SD
  
  // CONDITION level parameters (untransformed for non-centered parameterization)
  matrix[N_groups,N_cond_other] cond_delta_pr;      // drift rate head/emo condition effect
  
  //SUBJECT-level parameters (untransformed for non-centered parameterization)
  vector[N_subj] sub_alpha_pr;                       // threshold sep. subject mean
  vector[N_subj] sub_beta_pr;                        // start point subject mean
  matrix[N_subj,N_cond_other] sub_delta_present_pr;  // drift rate subject mean, stim present
  matrix[N_subj,N_cond_other] sub_delta_absent_pr;   // drift rate subject mean, stim absent
  vector[N_subj] sub_ndt_pr;                         // non-decision time subject mean
  vector<lower=0>[N_subj] sig_sub_delta_pr;          // trial level variability in drift rate for each subject
}

transformed parameters { 
  
  // SUBJECT-level transformed pars for non-centered parameterization
  vector<lower=0.1,upper=4>[N_subj] sub_alpha;              
  vector<lower=0,upper=1>[N_subj] sub_beta;                
  matrix<lower=-5,upper=5>[N_subj,N_cond_other] sub_delta_present;      
  matrix<lower=-5,upper=5>[N_subj,N_cond_other] sub_delta_absent;       
  vector<lower=0,upper=max(minRT)*0.98>[N_subj] sub_ndt;  

  for (i in 1:N_subj){
    sub_alpha[i] = 0.1 + 3.9 * Phi(mu_grp_alpha_pr[subj_group[i]] + sig_grp_alpha_pr[subj_group[i]] * sub_alpha_pr[i]); 
    sub_beta[i] = Phi(mu_grp_beta_pr[subj_group[i]] + sig_grp_beta_pr[subj_group[i]] * sub_beta_pr[i]);
    
    if(fix_ndt==1){
      sub_ndt[i] = 0.2;
    }else{
      sub_ndt[i] = (minRT[i]*0.98) * Phi(mu_grp_ndt_pr[subj_group[i]] + sig_grp_ndt_pr[subj_group[i]] * sub_ndt_pr[i]);
    }
    
   for (j in 1:N_cond_other){
     sub_delta_present[i,j] = -5 + 10 * Phi(mu_grp_delta_present_pr[subj_group[i]] + sig_grp_delta_pr[subj_group[i]] * sub_delta_present_pr[i,j] + cond_delta_pr[subj_group[i],j]);
     sub_delta_absent[i,j] = -5 + 10 *  Phi(mu_grp_delta_absent_pr[subj_group[i]] + sig_grp_delta_pr[subj_group[i]] * sub_delta_absent_pr[i,j] + cond_delta_pr[subj_group[i],j]);
    }
  }
}

model {
  
  // GROUP-level priors 
  mu_grp_alpha_pr ~ normal(0, 1);          
  mu_grp_beta_pr ~ normal(0, 1);          
  mu_grp_delta_present_pr ~ normal(0, 1);  
  mu_grp_delta_absent_pr ~ normal(0, 1);  
  mu_grp_ndt_pr ~ normal(0, 1);           
  
  for (i in 1:N_groups){
    sig_grp_alpha_pr[i] ~ normal(0,.2)T[0,];        
    sig_grp_beta_pr[i] ~ normal(0,.2)T[0,];        
    sig_grp_ndt_pr[i] ~ normal(0,.2)T[0,];          
    sig_grp_delta_pr[i] ~ normal(0,.2)T[0,];       
  }
  
  for(i in 1:N_subj){
    sig_sub_delta_pr[i]~ normal(0,.2)T[0,];
  }
  
  // CONDITION-level priors 
  to_vector(cond_delta_pr) ~ normal(0, 1);
  
  //SUBJECT-level priors
  sub_alpha_pr ~ normal(0, 1);             
  sub_beta_pr ~ normal(0, 1);             
  to_vector(sub_delta_present_pr) ~ normal(0, 1);    
  to_vector(sub_delta_absent_pr) ~ normal(0, 1);    
  sub_ndt_pr  ~ normal(0, 1);           
  
  for (i in 1:N_obs){ 
    if (cond[i]==1){ // if gaze signal = present
      if(choice[i]==1){ //if response = YES (correct)
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_present[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
      } else { //if response is NO (incorrect)
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_present[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
      }
    } else { // if gaze signal = absent 
      if(choice[i]==1){ // if response = YES (incorrect)
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_absent[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
      } else { // if response = NO (correct)
        RT[i] ~ wiener(sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_absent[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
      }
    }
  }
}

generated quantities {
  vector[N_obs] log_lik = rep_vector(0, N_obs); // log liklihood for each observation
  
  // GROUP-level transformed parameters
  vector<lower=0.1,upper=4>[N_groups] mu_alpha = 0.1 + 3.9*Phi(mu_grp_alpha_pr);         
  vector<lower=0,upper=1>[N_groups] mu_beta = Phi(mu_grp_beta_pr);                   
  vector <lower=0, upper=0.98>[N_groups] mu_ndt= 0.98*Phi(mu_grp_ndt_pr); 
  matrix<lower=-5,upper=5>[N_groups,N_cond_other] mu_delta_present_cond;
  matrix<lower=-5,upper=5>[N_groups,N_cond_other] mu_delta_absent_cond;
  
  for (i in 1:N_groups){
    for (j in 1:N_cond_other){
      mu_delta_present_cond[i,j] = -5 + 10*Phi(mu_grp_delta_present_pr[i]+cond_delta_pr[i,j]);
      mu_delta_absent_cond[i,j] = -5 + 10*Phi(mu_grp_delta_absent_pr[i]+cond_delta_pr[i,j]);
    }
  }

  // calculate log_lik for observation
  if(get_log_lik==1){
    for (i in 1:N_obs){
      if(cond[i]==1){ // if gaze signal = present
        if(choice[i]==1){ // if response = YES (correct)
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_present[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
        } else { // if response = NO (incorrect)
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_present[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
        }
      } else{ // if gaze signal = absent
        if(choice[i]==1){ // if response = YES (incorrect)
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], sub_beta[subj[i]], sub_delta_absent[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
        } else { // if response = NO (correct)
          log_lik[i] += wiener_lpdf(RT[i] | sub_alpha[subj[i]], sub_ndt[subj[i]], 1-sub_beta[subj[i]], -sub_delta_absent[subj[i],cond_other[i]],sig_sub_delta_pr[subj[i]],0.0,0.0);
        }
      }
    }
  }
}
