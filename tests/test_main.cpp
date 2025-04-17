#include <gtest/gtest.h>
#include "ttie/ttie.h"

using namespace ttie;

TEST(TensorTest, UninitializedTensor) {
    Tensor x;
    EXPECT_TRUE(x.shape.empty());
}

TEST(TensorTest, ValidateShape) {
    Tensor t;
    t.shape = {2, 3};
    EXPECT_TRUE(t.validate_shape());

    t.shape = {0, 3};
    EXPECT_FALSE(t.validate_shape());

    t.shape = {};
    EXPECT_FALSE(t.validate_shape());

    EXPECT_THROW(t.resize(), std::exception);
}

TEST(TensorTest, InitializeAndSetData) {
    Tensor t;
    t.shape = {2, 3};
    t.resize();
    
    EXPECT_EQ(t.data.size(), 6);
}

TEST(TensorTest, GradientOperations) {
    Tensor t;
    t.shape = {2, 3};
    t.resize();
    t.resize_grad();
    
    EXPECT_EQ(t.grad.size(), 6);
    
    for (size_t i = 0; i < t.grad.size(); ++i) {
        t.grad[i] = static_cast<float>(i);
    }
    
    t.zero_grad();
    
    for (size_t i = 0; i < t.grad.size(); ++i) {
        EXPECT_FLOAT_EQ(t.grad[i], 0.0f);
    }
}

TEST(LayerTest, ReLU) {
    ReLU relu;
    EXPECT_EQ(relu.parameters().size(), 0);

    Tensor input;
    input.shape = {2, 3};
    input.resize();
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = static_cast<float>(i) - 2.0f;
    }

    Tensor output;
    relu.forward(input, output);
    
    for (size_t i = 0; i < output.data.size(); ++i) {
        EXPECT_FLOAT_EQ(output.data[i], std::max(0.0f, input.data[i]));
    }

    output.resize_grad();
    for (size_t i = 0; i < output.grad.size(); ++i) {
        output.grad[i] = 1.0f;
    }
    
    relu.backward(output, input);
    
    for (size_t i = 0; i < input.grad.size(); ++i) {
        EXPECT_FLOAT_EQ(input.grad[i], input.data[i] > 0 ? 1.0f : 0.0f);
    }
}

TEST(LayerTest, Linear) {
    Linear linear(3, 2);
    
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2); 

    EXPECT_EQ(linear.weight.shape.size(), 2);
    EXPECT_EQ(linear.weight.shape[0], 3);
    EXPECT_EQ(linear.weight.shape[1], 2);
    EXPECT_EQ(linear.bias.shape.size(), 1);
    EXPECT_EQ(linear.bias.shape[0], 2);
    
    Tensor input;
    input.shape = {2, 3};
    input.resize();
    for (size_t i = 0; i < input.data.size(); ++i) {
        input.data[i] = static_cast<float>(i) / input.data.size();
    }
    
    Tensor output;
    linear.forward(input, output);
    
    EXPECT_EQ(output.shape.size(), 2);
    EXPECT_EQ(output.shape[0], 2);
    EXPECT_EQ(output.shape[1], 2);
    
    output.resize_grad();
    for (size_t i = 0; i < output.grad.size(); ++i) {
        output.grad[i] = 1.0f;
    }
    
    input.resize_grad();
    input.zero_grad();
    
    linear.backward(output, input);
    
    bool has_nonzero = false;
    for (size_t i = 0; i < input.grad.size(); ++i) {
        if (input.grad[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
    
    has_nonzero = false;
    for (size_t i = 0; i < linear.weight.grad.size(); ++i) {
        if (linear.weight.grad[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
    
    has_nonzero = false;
    for (size_t i = 0; i < linear.bias.grad.size(); ++i) {
        if (linear.bias.grad[i] != 0.0f) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

TEST(LayerTest, LinearVSTorch) {
    /* PyTorch reference forward and backward with predefined weights and biases
    import torch

    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    bias = torch.tensor([0.1, 0.2], requires_grad=True)

    output = torch.addmm(bias, input, weight)
    output.backward(torch.ones_like(output))

    # Print results
    print("PyTorch Output:", output)
    print("PyTorch Output Grad:", torch.ones_like(output))
    print("PyTorch Input Grad:", input.grad)
    print("PyTorch Weight Grad:", weight.grad)
    print("PyTorch Bias Grad:", bias.grad)

    # Output:
    PyTorch Output: tensor([[0.3200, 0.4800],
        [0.5900, 0.8400]], grad_fn=<AddmmBackward0>)
    PyTorch Output Grad: tensor([[1., 1.],
            [1., 1.]])
    PyTorch Input Grad: tensor([[0.3000, 0.7000, 1.1000],
            [0.3000, 0.7000, 1.1000]])
    PyTorch Weight Grad: tensor([[0.5000, 0.5000],
            [0.7000, 0.7000],
            [0.9000, 0.9000]])
    PyTorch Bias Grad: tensor([2., 2.])
    */

    Linear linear(3, 2);
    linear.weight.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    linear.bias.data = {0.1f, 0.2f};
    
    Tensor input;
    input.shape = {2, 3};
    input.resize();
    input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    input.resize_grad();
    
    // Forward pass
    Tensor output;
    linear.forward(input, output);
    
    // Check output values match PyTorch
    ASSERT_EQ(output.shape.size(), 2);
    ASSERT_EQ(output.shape[0], 2);
    ASSERT_EQ(output.shape[1], 2);
    EXPECT_NEAR(output.data[0], 0.32f, 1e-5f);
    EXPECT_NEAR(output.data[1], 0.48f, 1e-5f);
    EXPECT_NEAR(output.data[2], 0.59f, 1e-5f);
    EXPECT_NEAR(output.data[3], 0.84f, 1e-5f);
    
    // Backward pass
    output.resize_grad();
    output.grad = {1.0f, 1.0f, 1.0f, 1.0f};
    
    linear.backward(output, input);
    
    // Check input gradients match PyTorch
    ASSERT_EQ(input.grad.size(), 6);
    EXPECT_NEAR(input.grad[0], 0.3f, 1e-5f);
    EXPECT_NEAR(input.grad[1], 0.7f, 1e-5f);
    EXPECT_NEAR(input.grad[2], 1.1f, 1e-5f);
    EXPECT_NEAR(input.grad[3], 0.3f, 1e-5f);
    EXPECT_NEAR(input.grad[4], 0.7f, 1e-5f);
    EXPECT_NEAR(input.grad[5], 1.1f, 1e-5f);
    
    // Check weight gradients match PyTorch
    ASSERT_EQ(linear.weight.grad.size(), 6);
    EXPECT_NEAR(linear.weight.grad[0], 0.5f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[1], 0.5f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[2], 0.7f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[3], 0.7f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[4], 0.9f, 1e-5f);
    EXPECT_NEAR(linear.weight.grad[5], 0.9f, 1e-5f);
    
    // Check bias gradients match PyTorch
    ASSERT_EQ(linear.bias.grad.size(), 2);
    EXPECT_NEAR(linear.bias.grad[0], 2.0f, 1e-5f);
    EXPECT_NEAR(linear.bias.grad[1], 2.0f, 1e-5f);
}

TEST(ModelTest, ForwardAndBackwardVSTorch) {
    /* Pytorch reference 
    import torch

    input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    weight = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True)
    bias = torch.tensor([0.1, 0.2], requires_grad=True)

    # First addmm operation
    output1 = torch.addmm(bias, input, weight)
    relu_output = torch.relu(output1)

    # Define weights and bias for the second addmm (2->1)
    weight2 = torch.tensor([[0.7], [0.8]], requires_grad=True)
    bias2 = torch.tensor([0.3], requires_grad=True)

    output = torch.addmm(bias2, relu_output, weight2)
    output.backward(torch.ones_like(output))

    # Print results
    print("PyTorch First Output:", output1)
    print("PyTorch ReLU Output:", relu_output)
    print("PyTorch Final Output:", output)
    print("PyTorch Output Grad:", torch.ones_like(output))
    print("PyTorch Input Grad:", input.grad)
    print("PyTorch Weight Grad:", weight.grad)
    print("PyTorch Bias Grad:", bias.grad)
    print("PyTorch Weight2 Grad:", weight2.grad)
    print("PyTorch Bias2 Grad:", bias2.grad)

    Output:
    PyTorch First Output: tensor([[0.3200, 0.4800],
        [0.5900, 0.8400]], grad_fn=<AddmmBackward0>)
    PyTorch ReLU Output: tensor([[0.3200, 0.4800],
            [0.5900, 0.8400]], grad_fn=<ReluBackward0>)
    PyTorch Final Output: tensor([[0.9080],
            [1.3850]], grad_fn=<AddmmBackward0>)
    PyTorch Output Grad: tensor([[1.],
            [1.]])
    PyTorch Input Grad: tensor([[0.2300, 0.5300, 0.8300],
            [0.2300, 0.5300, 0.8300]])
    PyTorch Weight Grad: tensor([[0.3500, 0.4000],
            [0.4900, 0.5600],
            [0.6300, 0.7200]])
    PyTorch Bias Grad: tensor([1.4000, 1.6000])
    PyTorch Weight2 Grad: tensor([[0.9100],
            [1.3200]])
    PyTorch Bias2 Grad: tensor([2.])
    */
    
    Model model;
    model.add_layer(new Linear(3, 2));
    model.add_layer(new ReLU());
    model.add_layer(new Linear(2, 1));
    
    // Initialize with the same values as in PyTorch reference
    Linear* layer1 = static_cast<Linear*>(model.layers[0]);
    layer1->weight.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    layer1->bias.data = {0.1f, 0.2f};
    
    Linear* layer2 = static_cast<Linear*>(model.layers[2]);
    layer2->weight.data = {0.7f, 0.8f};
    layer2->bias.data = {0.3f};
    
    // Prepare input
    Tensor input;
    input.shape = {2, 3};
    input.resize();
    input.data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    input.resize_grad();
    
    // Forward pass
    Tensor output;
    model.forward(input, output);
    
    // Verify final output shape and values
    ASSERT_EQ(output.shape.size(), 2);
    ASSERT_EQ(output.shape[0], 2);
    ASSERT_EQ(output.shape[1], 1);
    EXPECT_NEAR(output.data[0], 0.9080f, 1e-4f);
    EXPECT_NEAR(output.data[1], 1.3850f, 1e-4f);
    
    // Backward pass
    output.resize_grad();
    output.grad = {1.0f, 1.0f};
    
    model.backward(output, input);
    
    // Check input gradients
    ASSERT_EQ(input.grad.size(), 6);
    EXPECT_NEAR(input.grad[0], 0.2300f, 1e-4f);
    EXPECT_NEAR(input.grad[1], 0.5300f, 1e-4f);
    EXPECT_NEAR(input.grad[2], 0.8300f, 1e-4f);
    EXPECT_NEAR(input.grad[3], 0.2300f, 1e-4f);
    EXPECT_NEAR(input.grad[4], 0.5300f, 1e-4f);
    EXPECT_NEAR(input.grad[5], 0.8300f, 1e-4f);
    
    // Check layer1 weight gradients
    ASSERT_EQ(layer1->weight.grad.size(), 6);
    EXPECT_NEAR(layer1->weight.grad[0], 0.3500f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[1], 0.4000f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[2], 0.4900f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[3], 0.5600f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[4], 0.6300f, 1e-4f);
    EXPECT_NEAR(layer1->weight.grad[5], 0.7200f, 1e-4f);
    
    // Check layer1 bias gradients
    ASSERT_EQ(layer1->bias.grad.size(), 2);
    EXPECT_NEAR(layer1->bias.grad[0], 1.4000f, 1e-4f);
    EXPECT_NEAR(layer1->bias.grad[1], 1.6000f, 1e-4f);
    
    // Check layer2 weight gradients
    ASSERT_EQ(layer2->weight.grad.size(), 2);
    EXPECT_NEAR(layer2->weight.grad[0], 0.9100f, 1e-4f);
    EXPECT_NEAR(layer2->weight.grad[1], 1.3200f, 1e-4f);
    
    // Check layer2 bias gradients
    ASSERT_EQ(layer2->bias.grad.size(), 1);
    EXPECT_NEAR(layer2->bias.grad[0], 2.0000f, 1e-4f);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}