import tensorflow as tf

slim = tf.contrib.slim

# g-step에서는 classifier와 generator를 학습시킨다
def g_step_loss(source_images, source_labels, discriminator, classifier, num_classes):
    g_loss = 0

    # discriminator가 fake를 real이라고 말했을 때 작아지는 loss
    g_loss += tf.losses.sigmoid_cross_entropy(
        logits=discriminator['transferred_domain_logits'],
        multi_class_labels=tf.ones_like(discriminator['transferred_domain_logits']),
        weights=0.011)      # 논문에서 mnist-m 예제는 0.011을 썻다고함

    # 여기서 classification loss를 받는 것은 cls를 학습시키기 위해서가 아니라 gen의 로스를 backprop 받기 위해서다
    g_loss += classification_loss(classifier, source_labels, num_classes)

    return g_loss

def classification_loss(classifier, source_labels, num_classes):
    one_hot_labels = slim.one_hot_encoding(source_labels, num_classes)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=classifier['source_task_logits'],
        weights=0.01)

    loss += tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=classifier['transferred_task_logits'],
        weights=0.01)

    return loss

def discriminator_loss(discriminator):
    # discriminator는 fake를 받아서 0로로 맞춰야 loss가 줄어든다.
    transferred_domain_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(discriminator['transferred_domain_logits']),
        logits=discriminator['transferred_domain_logits'])

    # 또한 real을 받아서 1로 맞춰야 loss가 줄어든다.
    target_domain_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(discriminator['target_domain_logits']),
        logits=discriminator['target_domain_logits'])

    # 0.13은 논문의 수치. 논문 B.2를 확인
    total_domain_loss = (transferred_domain_loss + target_domain_loss) * 0.13
    return total_domain_loss

# d-step에서는 discriminator를 학습시킨다.
def d_step_loss(discriminator, classifier, source_labels, num_classes):
    # discriminator loss
    total_domain_loss = discriminator_loss(discriminator)
    # 여기서 classification을 진짜로 학습한다.
    cls_loss = classification_loss(classifier, source_labels, num_classes)

    return cls_loss + total_domain_loss