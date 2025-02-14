import tempfile

import pytest
import torch

from flowcon.transforms.reshape import FlattenTransform, SqueezeTransform
from tests.transforms.transform_test import TransformTest


class SqueezeTransformTest(TransformTest):
    def setUp(self):
        self.transform = SqueezeTransform()

    def test_forward(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                outputs, logabsdet = self.transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, c * 4, h // 2, w // 2])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_values(self):
        inputs = torch.arange(1, 17, 1).long().view(1, 1, 4, 4)
        outputs, _ = self.transform(inputs)

        def assert_channel_equal(channel, values):
            self.assertEqual(outputs[0, channel, ...], torch.LongTensor(values))

        assert_channel_equal(0, [[1, 3], [9, 11]])
        assert_channel_equal(1, [[2, 4], [10, 12]])
        assert_channel_equal(2, [[5, 7], [13, 15]])
        assert_channel_equal(3, [[6, 8], [14, 16]])

    def test_forward_wrong_shape(self):
        batch_size = 10
        for shape in [[32, 3, 3], [32, 5, 5], [32, 4]]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.transform(inputs)

    def test_forward_inverse_are_consistent(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                self.assert_forward_inverse_are_consistent(self.transform, inputs)

    def test_inverse_wrong_shape(self):
        batch_size = 10
        for shape in [[3, 4, 4], [33, 4, 4], [32, 4]]:
            with self.subTest(shape=shape):
                inputs = torch.randn(batch_size, *shape)
                with self.assertRaises(ValueError):
                    self.transform.inverse(inputs)


class FlattenTransformTest(TransformTest):
    def test_forward(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            transform = FlattenTransform()
            with self.subTest(shape=shape):
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, c * h * w])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_inverse(self):
        batch_size = 10
        for shape in [[32, 4, 4], [16, 8, 8]]:
            with self.subTest(shape=shape):
                c, h, w = shape
                dim = c * h * w
                inputs = torch.randn(batch_size, c, h, w)

                transform = FlattenTransform()

                with pytest.raises(RuntimeError):
                    transform.inverse(inputs.reshape(-1, dim))

                transform(inputs)
                input_rec, logabsdet = transform.inverse(inputs.reshape(-1, dim))
                self.assert_tensor_equal(input_rec, inputs)
                self.assert_tensor_equal(logabsdet, torch.zeros(batch_size))

    def test_immutable(self):
        batch_size = 10
        transform = FlattenTransform()
        with pytest.raises(RuntimeError):
            for shape in [[32, 4, 4], [16, 8, 8]]:
                c, h, w = shape
                inputs = torch.randn(batch_size, c, h, w)
                outputs, logabsdet = transform(inputs)
                self.assert_tensor_is_good(outputs, [batch_size, c * h * w])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.assertEqual(logabsdet, torch.zeros(batch_size))

    def test_forward_inverse_values(self):
        transform = FlattenTransform()
        inputs = torch.randn(10, 5, 3, 10)
        outputs, _ = transform(inputs)
        self.assert_tensor_equal(outputs, inputs.reshape(10, -1))
        self.assert_tensor_equal(transform.inverse(inputs.reshape(10, -1))[0], inputs)

        self.assert_forward_inverse_are_consistent(FlattenTransform(), inputs)

    @pytest.mark.expensive
    def test_compilable(self):
        transform = torch.compile(FlattenTransform(), mode="reduce-overhead")
        inputs = torch.randn(10, 5, 3, 10)
        outputs, _ = transform(inputs)
        self.assert_tensor_equal(outputs, inputs.reshape(10, -1))
        self.assert_tensor_equal(transform.inverse(inputs.reshape(10, -1))[0], inputs)

        self.assert_forward_inverse_are_consistent(FlattenTransform(), inputs)

        with pytest.raises(RuntimeError):
            transform(torch.randn(10, 5, 3, 15))

    def test_serialization(self):
        """Ensure cached shape and lock persist after saving/loading."""
        transform = FlattenTransform()
        x = torch.randn(4, 3, 32, 32)
        transform.forward(x)  # Sets _cached_shape and locks it

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=True) as f:
            torch.save(transform.state_dict(), f.name)

            # Create a new instance and load state
            new_transform = FlattenTransform()
            new_transform.load_state_dict(torch.load(f.name))

        # Check that the loaded instance has the correct state
        assert new_transform._cached_shape.tolist() == [3, 32, 32]
        assert new_transform._locked.item() == 1  # Should still be locked


if __name__ == "__main__":
    pytest.main()
