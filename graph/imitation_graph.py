"""
Imitating from an Expert
"""
import tensorflow as tf
import baselines.common.tf_util as U


def imit_build_act(
              make_obs_ph, 
              bnn_func, 
              num_actions,
              en,
              bnn_explore=0.01,
              scope="Imitation",
              reuse=None,              
              use_sign=False,):

    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        is_training = tf.placeholder(tf.bool, (), name='is_training')

        # construct BNNs
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
                         outputs=[tf.squeeze(bnn_output_actions), BNN_output],
                         givens={is_training:False},)

        return bnn_act, is_training

def imit_build_train(
                make_obs_ph, 
                bnn_func,
                learning_rate,
                num_actions, 
                en,

                # raw_input_ph is the placeholder accept the raw image            
                raw_input_ph,
                target_output,

                gamma,
                grad_norm_clipping=None,
                alpha=20,
                bnn_explore=0.01,
                scope="Imitation", 
                reuse=None,
                use_sign=False,
                ):

    bnn_act_f, is_training = imit_build_act(
                    make_obs_ph, bnn_func, num_actions, 
                    bnn_explore=bnn_explore,
                    en=en, scope=scope, reuse=reuse,
                    use_sign=use_sign,
                )
                

    with tf.variable_scope(scope, reuse=reuse):
        loss_list = []
        bnn_func_vars_list = []
        
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
            bnn_func_vars_list += bnn_func_vars

            loss_list.append(tf.reduce_mean(
                    tf.square(bnn_output - alpha * target_output_list[count])
                ))                
                
            #TODO
            predict = tf.argmax(bnn_output, axis=1)
            accu = tf.reduce_mean(tf.cast(tf.equal(predict, target_label_list[count]), "float"))
            accu_list.append(accu)

        total_loss = sum(loss_list)
        
        assert grad_norm_clipping is not None
        optimize_expr = U.minimize_and_clip(
                                        tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4),
                                        total_loss,
                                        var_list=bnn_func_vars_list,
                                        clip_val=grad_norm_clipping)        

        train = U.function(
            inputs=[raw_input_ph, is_training],
            # TODO
            outputs=accu_list + loss_list,
            updates=[optimize_expr],
            givens={is_training: True},
        )

    return bnn_act_f, train
