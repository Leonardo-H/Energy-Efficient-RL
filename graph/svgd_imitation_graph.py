import tensorflow as tf
import baselines.common.tf_util as U
import numpy as np


def svgd_imit_build_act(
              make_obs_ph, 
              bnn_func, 
              num_actions,
              en,
              bnn_explore=0.01,
              scope="SVGD_Imitation",
              reuse=None,
              use_sign=False):

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        is_training = tf.placeholder(tf.bool, (), name='is_training')
        
        BNN_output_list = []
        for count in range(en):
            bnn_output_tem = bnn_func(observations_ph.get(), num_actions,
                            is_training=is_training, scope="bnn_func" + str(count) + '_', 
                            use_sign=use_sign,
                            )
            BNN_output_list.append(bnn_output_tem)

        BNN_output = sum(BNN_output_list)

        eps = tf.constant(0.01)
        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random_bnn = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps

        bnn_deterministic_actions = tf.argmax(BNN_output, axis=1)
        bnn_output_actions = tf.where(chose_random_bnn, random_actions, bnn_deterministic_actions)
 
        bnn_act = U.function(inputs=[observations_ph, is_training],
                         outputs=[tf.squeeze(bnn_output_actions)],
                         givens={is_training:False},)

        return bnn_act, is_training

def svgd_imit_build_train(
                make_obs_ph, 
                bnn_func,
                learning_rate,
                num_actions, 
                en,
                
                raw_input_ph,
                target_output,
                
                gamma,
                grad_norm_clipping=None,
                alpha=20,
                bnn_explore=0.01,
                scope="SVGD_Imitation",
                reuse=None,
                use_sign=False,
                ):

    bnn_act_f, is_training = svgd_imit_build_act(
                    make_obs_ph, bnn_func, num_actions, 
                    bnn_explore=bnn_explore,
                    en=en, scope=scope, reuse=reuse,
                    use_sign=use_sign,
                )

    with tf.variable_scope(scope, reuse=reuse):
        
        bnn_func_trainable_vars_list = []
        bnn_func_trainable_vars_one_list = []

        log_p_list = []    
        

        obs_t = tf.cast(raw_input_ph, tf.float32) / 255.0
        obs_t_list = tf.split(obs_t, en, axis=0)

        target_output_list = tf.split(target_output, en, axis=0)

        # TODO
        accu_list = []
        target_label_list = tf.split(tf.argmax(target_output, axis=1), en, axis=0)

        for count in range(en):
            bnn_output = bnn_func(obs_t_list[count], num_actions, 
                                scope="bnn_func" + str(count) + '_',
                                is_training=is_training, reuse=True)

            bnn_func_vars = U.scope_vars(U.absolute_scope_name("bnn_func" + str(count) + '_')) 
            bnn_func_trainable_vars = []
            
            # Moving vars is non-trainable, we should pick them up
            for bv_t in bnn_func_vars:
                if not 'moving' in bv_t.name and 'Not_Reuse' not in bv_t.name:
                    bnn_func_trainable_vars.append(bv_t)

            # This one put all vars in one list
            bnn_func_trainable_vars_one_list += bnn_func_trainable_vars
            # This one just put the vars list into one list
            bnn_func_trainable_vars_list.append(bnn_func_trainable_vars)
            
            log_p_list.append(
                -tf.reduce_mean(
                    tf.square(bnn_output - alpha * target_output_list[count])
                )
            )

            #TODO
            predict = tf.argmax(bnn_output, axis=1)
            accu = tf.reduce_mean(tf.cast(tf.equal(predict, target_label_list[count]), "float"))
            accu_list.append(accu)

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

        na_t = tf.reduce_sum(tf.square(theta), 1)
        nb_t = tf.reduce_sum(tf.square(theta), 1)
        assert na_t.get_shape().as_list() == [en]
        assert nb_t.get_shape().as_list() == [en]

        na = tf.reshape(na_t, [-1, 1])
        nb = tf.reshape(nb_t, [1, -1])
        assert na.get_shape().as_list() == [en, 1]
        assert nb.get_shape().as_list() == [1, en]

        D = tf.maximum(0.0, na - 2 * tf.matmul(theta, theta, False, True) + nb)
        assert D.get_shape().as_list() == [en, en]

        # step 3 kernel
        print('step 3 kernel')
        if en == 2:
            # when 2 particles, use percentile may return 0
            D_mid = (tf.reduce_max(D) + tf.reduce_min(D)) / 2
        else:
            D_mid = tf.contrib.distributions.percentile(D, 50)
        
        # h**2 = mid / log(en + 1) / 100
        h_tau = tf.placeholder(shape=(), dtype=tf.float32, name='h_tau')
        h = tf.sqrt(0.5 * D_mid / tf.log(en + 1.0)) * h_tau
        kernel = tf.exp( -D / h ** 2 / 2)
        
        assert kernel.get_shape().as_list() == [en, en], 'kernel shape should be (en, en)'

        # step 4: kernel gradients
        print('step 4 kernel gradients')
        dxkxy = -tf.matmul(kernel, theta)
        sumkxy = tf.expand_dims(tf.reduce_sum(kernel, axis=1), axis=1)

        assert dxkxy.get_shape().as_list() == [en, variables_num]
        assert sumkxy.get_shape().as_list() == [en, 1]

        dxkxy += sumkxy * theta
        dxkxy /= (h ** 2)

        assert dxkxy.get_shape().as_list() == [en, variables_num]

        # step 5: log_p gradients
        print('step 5: log_p gradients')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)
        log_p_sum = sum(log_p_list)
        grad_log_p = optimizer.compute_gradients(
                            log_p_sum, 
                            var_list=bnn_func_trainable_vars_one_list,
                        )
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

        # what we calculate is SVGD grad, if we want to use adam, we should change their sign        
        tau = tf.placeholder(shape=(), dtype=tf.float32, name='decay_tau')
        grad_1 = -dxkxy * tau
        grad_2 = -tf.matmul(kernel, grad_matrix)
        gradients = (grad_1 + grad_2) / en

        gradients_flatten = tf.reshape(gradients, shape=[-1])

        # step 6 apply gradients, pick up their grads one by one
        print('step 6 apply gradients')
        start = 0
        for i, (grad, var) in enumerate(grad_log_p):
            grad_flatten = gradients_flatten[start:start + vars_num_list[i]]
            clipped_grad = tf.clip_by_norm(
                tf.reshape(grad_flatten, vars_shape_list[i]), grad_norm_clipping
            )
            grad_log_p[i] = (clipped_grad, var)
            start += vars_num_list[i]
        assert start == variables_num * en, 'expect start is {}, but it is {}'.format(variables_num ,start)

        optimize_expr = optimizer.apply_gradients(grad_log_p)
        
        train = U.function(
            inputs=[raw_input_ph, tau, h_tau, is_training],
            outputs=accu_list + log_p_list + [D, D_mid, h, kernel],
            updates=[optimize_expr],
            givens={
                is_training: True,                
            },
        )

    return bnn_act_f, train
