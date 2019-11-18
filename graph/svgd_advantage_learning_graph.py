import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np

def svgd_adv_build_act(make_obs_ph, 
              adv_func, 
              num_actions,
              en=1,
              scope="svgd_advantage_learning",
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
                            is_training=is_training, scope="adv_func" + str(count) + '_', 
                            )
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
                         outputs=[tf.squeeze(output_actions), a_values] + a_values_list,
                         givens={update_eps_ph: -1.0, stochastic_ph: True, is_training:False},
                         updates=[update_eps_expr])

        return act, is_training

def svgd_adv_build_train(make_obs_ph, 
                v_func, 
                adv_func,
                num_actions, 
                learning_rate,
                en=1,
                grad_norm_clipping=None, 
                gamma=1.0,   
                scope="svgd_advantage_learning", 
                reuse=None,
                ):
                
                
    act_f, is_training = svgd_adv_build_act(make_obs_ph, adv_func, num_actions, 
                                   en=en, scope=scope, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):        
        # for advantage
        adv_values_list = []
        target_adv_func_vars_list = []

        weighted_error_list = []

        # construct placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")

        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        rew_t_ph_list = tf.split(rew_t_ph, en, axis=0)

        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        obs_tp1_list = tf.split(obs_tp1_input.get(), en, axis=0)
        
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        done_mask_ph_list = tf.split(done_mask_ph, en, axis=0)
        
        obs_t_input_list = tf.split(obs_t_input.get(), en, axis=0)
        act_t_ph_list = tf.split(act_t_ph, en, axis=0)

        # construct v function
        v_t = tf.squeeze(v_func(obs_t_input.get(), scope="v_func", reuse=False))
        v_t_list = tf.split(v_t, en, axis=0)
        v_func_vars = U.scope_vars(U.absolute_scope_name("v_func"))

        v_tp1 = tf.squeeze(v_func(obs_tp1_input.get(), scope="target_v_func", reuse=False))
        target_v_func_vars = U.scope_vars(U.absolute_scope_name("target_v_func"))
        v_tp1_list = tf.split(v_tp1, en, axis=0)

        target_adv_values_list = []
        q_t_selected_list = []
        q_t_selected_target_list = []
        
        bnn_func_trainable_vars_list = []
        bnn_func_trainable_vars_one_list = []
        bnn_func_all_vars_list = []
        log_p_list = []
        
        
        for count in range(en):            
            adv_t = adv_func(obs_t_input_list[count], num_actions, is_training=is_training, 
                        scope="adv_func" + str(count) + '_', reuse=True,
                        )  # reuse parameters from act

            # collect all variables and divide moving variables
            bnn_func_vars = U.scope_vars(U.absolute_scope_name("adv_func" + str(count) + '_')) 
            bnn_func_trainable_vars = []
            for bv_t in bnn_func_vars:
                if 'moving' not in bv_t.name:
                    bnn_func_trainable_vars.append(bv_t)

            bnn_func_trainable_vars_list.append(bnn_func_trainable_vars)
            bnn_func_trainable_vars_one_list += bnn_func_trainable_vars
            
            bnn_func_all_vars_list += bnn_func_vars
            
            # target network
            adv_tp1 = adv_func(obs_tp1_list[count], num_actions, is_training=False,
                        scope="target_adv_func" + str(count) + '_',
                        )                            
            target_adv_func_vars_list += U.scope_vars(U.absolute_scope_name("target_adv_func" + str(count) + '_'))

            target_adv_values_list.append(adv_tp1)

            adv_tp1_best = tf.reduce_max(adv_tp1, 1)    
            
            q_tp1_best = v_tp1_list[count] + adv_tp1_best
            q_tp1_best_masked = (1.0 - done_mask_ph_list[count]) * q_tp1_best
            q_t_selected_target = rew_t_ph_list[count] + gamma * q_tp1_best_masked

            q_t_selected_target_list.append(q_t_selected_target)


            # q scores for actions which we know were selected in the given state.
            adv_t_selected = tf.reduce_sum(adv_t * tf.one_hot(act_t_ph_list[count], num_actions), 1)

            # compute estimate of best possible value starting from state at t + 1                
            q_t_selected = v_t_list[count] + adv_t_selected  

            # compute the error (potentially clipped)
            weighted_error_list.append(
                tf.reduce_mean(
                    tf.square(
                        v_t_list[count] + tf.stop_gradient(adv_t_selected - q_t_selected_target_list[count])
                    )
                )
            )
            log_p_list.append(
                -tf.reduce_mean(
                    tf.square(
                        adv_t_selected + tf.stop_gradient(v_t_list[count] - q_t_selected_target_list[count])
                    )
                )
            )
            adv_values_list.append(adv_t)
            q_t_selected_list.append(q_t_selected)

        all_vars_list = v_func_vars + bnn_func_all_vars_list
        all_target_vars_list = target_v_func_vars + target_adv_func_vars_list

        update_target_expr = []

        for var, var_target in zip(sorted(all_vars_list, key=lambda v: v.name),
                                sorted(all_target_vars_list, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        """
        Use SVGD
        """
        # use svgd
        # step 1: flatten all variables
        print('step 1: flatten all variables')
        bnn_func_trainable_vars_flatten_list = []
        for index in range(len(bnn_func_trainable_vars_list)):
            bnn_func_trainable_vars = bnn_func_trainable_vars_list[index]
            bnn_func_trainable_vars_flatten = tf.concat(
                [tf.reshape(var, shape=[-1]) for var in bnn_func_trainable_vars], axis=0
            )
            # here the shape should be (variables_num)
            bnn_func_trainable_vars_flatten_list.append(
                tf.expand_dims(bnn_func_trainable_vars_flatten, axis=0)
            )
        variables_num = bnn_func_trainable_vars_flatten.get_shape().as_list()[0]
        print('We have total {} variables'.format(variables_num))

        # step 2: pairwise distance
        print('step 2: pairwise distance')
        # here the shape should be (en, variables_num)
        theta = tf.concat(bnn_func_trainable_vars_flatten_list, axis=0)
        
        assert theta.get_shape().as_list() == [en, variables_num]

        na = tf.reduce_sum(tf.square(theta), 1)
        nb = tf.reduce_sum(tf.square(theta), 1)
        assert na.get_shape().as_list() == [en]
        assert nb.get_shape().as_list() == [en]

        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        assert na.get_shape().as_list() == [en, 1]
        assert nb.get_shape().as_list() == [1, en]

        D = tf.maximum(0.0, na - 2 * tf.matmul(theta, theta, False, True) + nb)
        assert D.get_shape().as_list() == [en, en]

        # step 3 kernel
        print('step 3 kernel')
        D_mid = tf.contrib.distributions.percentile(D, 50)
        h_tau = tf.placeholder(shape=(), dtype=tf.float32, name='h_tau')
        h = tf.sqrt(0.5 * D_mid / tf.log(en + 1.0)) * h_tau
        kernel = tf.exp( -D / h ** 2 / 2)
        
        assert kernel.get_shape().as_list() == [en, en], 'kernel shape should be (en, en)'

        # step 4 kernel gradients
        print('step 4 kernel gradients')
        dxkxy = 0 - tf.matmul(kernel, theta)
        sumkxy = tf.expand_dims(tf.reduce_sum(kernel, axis=1), axis=1)

        assert dxkxy.get_shape().as_list() == [en, variables_num]
        assert sumkxy.get_shape().as_list() == [en, 1]

        dxkxy += sumkxy * theta
        dxkxy /= (h ** 2)

        assert dxkxy.get_shape().as_list() == [en, variables_num]

        # step 5: log_p gradients
        print('step 5: log_p gradients')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)

        # V function parameters' loss function
        v_loss = sum(weighted_error_list)
        # BNN function parameters' loss function
        log_p_sum = sum(log_p_list)
        total_loss = v_loss + log_p_sum
        
        total_grads = optimizer.compute_gradients(
                            total_loss, 
                            var_list=v_func_vars + bnn_func_trainable_vars_one_list,
                        )
        grad_v_loss = total_grads[0:len(v_func_vars)]
        grad_log_p = total_grads[len(v_func_vars):]
        assert len(grad_v_loss) == len(v_func_vars)
        assert len(grad_log_p) == len(bnn_func_trainable_vars_one_list)

        grad_list = []
        vars_shape_list = []
        vars_num_list = []

        for i, (grad, var) in enumerate(grad_log_p):
            assert grad is not None
            vars_shape = grad.get_shape().as_list()
            vars_shape_list.append(vars_shape)
            vars_num_list.append(np.prod(vars_shape))
            grad_list.append(grad)
            
        grad_flatten_concat = tf.concat([tf.reshape(gv, shape=[-1]) for gv in grad_list], axis=0)
        grad_matrix = tf.reshape(grad_flatten_concat, [en, -1])
        assert grad_matrix.get_shape().as_list() == [en, variables_num]
    
        # temperature
        tau = tf.placeholder(shape=(), dtype=tf.float32, name='decay_tau')
        grad_1 = -dxkxy * tau
        grad_2 = -tf.matmul(kernel, grad_matrix)
        gradients = grad_1 + grad_2

        gradients_flatten = tf.reshape(gradients, shape=[-1])

        # step 6 apply gradients
        print('step 6 apply gradients')
        start = 0
        for i, (grad, var) in enumerate(grad_log_p):
            assert grad is not None
            grad_flatten = gradients_flatten[start:start + vars_num_list[i]]
            clipped_grad = tf.clip_by_norm(
                tf.reshape(grad_flatten, vars_shape_list[i]), grad_norm_clipping
            )
            grad_log_p[i] = (clipped_grad, var)
            start += vars_num_list[i]

        for i, (grad, var) in enumerate(grad_v_loss):
            assert grad is not None
            clipped_grad = tf.clip_by_norm(
                grad, grad_norm_clipping
            )
            grad_v_loss[i] = (clipped_grad, var)

        assert start == variables_num * en, 'expect start is {}, but it is {}'.format(variables_num ,start)

        optimize_expr = optimizer.apply_gradients(grad_v_loss + grad_log_p)

        other_output = [D, D_mid, h, kernel]
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                tau,
                h_tau,
                is_training,
            ],
            outputs=weighted_error_list + other_output,
            updates=[optimize_expr],
            givens={is_training:True}
        )
        update_target = U.function([], [], updates=[update_target_expr])

        predict_values_outputs = [v_t, v_tp1] + adv_values_list +\
                                 target_adv_values_list + q_t_selected_list + q_t_selected_target_list
        
        predict_values = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                is_training,
            ],
            outputs=predict_values_outputs,
            givens={is_training:False}
        )

    return act_f, train, update_target, {'predict_values': predict_values}
