"""
Advantage Learning Graph
"""
import tensorflow as tf
import baselines.common.tf_util as U


def adv_build_act(make_obs_ph, 
              adv_func, 
              num_actions,
              en,
              scope="advantage_learning",
              reuse=None,
              ):
    
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        is_training = tf.placeholder(tf.bool, (), name='is_training')
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        a_values_list = []

        for count in range(en):
            adv_tem = adv_func(observations_ph.get(), num_actions, 
                            is_training=is_training, scope="adv_func" + str(count) + '_',)
            a_values_list.append(adv_tem)

        a_values = sum(a_values_list)
        deterministic_actions = tf.argmax(a_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph, is_training],
                        #only outputs=[tf.squeeze(output_actions)] is enough
                         outputs=[tf.squeeze(output_actions), a_values],
                         givens={update_eps_ph: -1.0, stochastic_ph: True, is_training:False},
                         updates=[update_eps_expr])

        return act, is_training

def adv_build_train(make_obs_ph, 
                v_func, 
                adv_func,
                num_actions,
                learning_rate,
                en,
                grad_norm_clipping=None,
                gamma=0.99,
                scope="advantage_learning", 
                reuse=None,
                ):
                
                
    act_f, is_training = adv_build_act(make_obs_ph, adv_func, num_actions, 
                                   en=en, scope=scope, reuse=reuse,)

    with tf.variable_scope(scope, reuse=reuse):
        
        adv_func_vars_list = []
        target_adv_func_vars_list = []        
        error_list = []

        # construct placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        
        obs_t_input_list = tf.split(obs_t_input.get(), en, axis=0)
        act_t_ph_list = tf.split(act_t_ph, en, axis=0)
        rew_t_ph_list = tf.split(rew_t_ph, en, axis=0)
        obs_tp1_input_list = tf.split(obs_tp1_input.get(), en, axis=0)
        done_mask_ph_list = tf.split(done_mask_ph, en, axis=0)

        # build v function
        v_t = tf.squeeze(v_func(obs_t_input.get(), scope="v_func", reuse=False))
        v_t_list = tf.split(v_t, en, axis=0)
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))

        # build v target
        v_tp1 = tf.squeeze(v_func(obs_tp1_input.get(), scope="target_v_func", reuse=False))
        v_tp1_list = tf.split(v_tp1, en, axis=0)
        target_v_func_vars = U.scope_vars(U.absolute_scope_name("target_v_func"))

        
        for count in range(en):
            # build BNN
            adv_t = adv_func(obs_t_input_list[count], num_actions, is_training=is_training, 
                        scope="adv_func" + str(count) + '_', reuse=True,
                        )

            adv_func_vars = U.scope_vars(U.absolute_scope_name("adv_func" + str(count) + '_'))
            adv_func_vars_list += adv_func_vars
            
            # build BNN target
            adv_tp1 = adv_func(obs_tp1_input_list[count], num_actions, is_training=False,
                        scope="target_adv_func" + str(count) + '_',
                        )
            target_adv_func_vars_list += U.scope_vars(U.absolute_scope_name("target_adv_func" + str(count) + '_'))

            adv_t_selected = tf.reduce_sum(adv_t * tf.one_hot(act_t_ph_list[count], num_actions), 1)

            adv_tp1_best = tf.reduce_max(adv_tp1, 1)

            q_t_selected = v_t_list[count] + adv_t_selected
            q_tp1_best = v_tp1_list[count] + adv_tp1_best
            q_tp1_best_masked = (1.0 - done_mask_ph_list[count]) * q_tp1_best
            q_t_selected_target = rew_t_ph_list[count] + gamma * q_tp1_best_masked

            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            
            errors = tf.reduce_mean(tf.square(td_error))
            error_list.append(errors)

        all_vars_list = v_func_vars + adv_func_vars_list
        all_target_vars_list = target_v_func_vars + target_adv_func_vars_list

        total_loss = sum(error_list)
        
        assert grad_norm_clipping is not None
        optimize_expr = U.minimize_and_clip(
                                            tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4),
                                            total_loss,
                                            var_list=all_vars_list,
                                            clip_val=grad_norm_clipping
                                        )
        update_target_expr = []

        for var, var_target in zip(sorted(all_vars_list, key=lambda v: v.name),
                                sorted(all_target_vars_list, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                is_training,
            ],
            outputs=error_list,
            updates=[optimize_expr],
            givens={is_training:True}
        )
        update_target = U.function([], [], updates=[update_target_expr])

    return act_f, train, update_target