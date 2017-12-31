import Augmentor
p = Augmentor.Pipeline("two",save_format=u'BMP')

p.rotate(probability=0.7, max_left_rotation=8, max_right_rotation=8)
p.rotate90(probability=0.7)
p.rotate270(probability=0.7)

p.sample(1500)
