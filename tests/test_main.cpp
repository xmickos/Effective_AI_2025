#include <gtest/gtest.h>
#include "ttie/ttie.h"

using namespace ttie;

TEST(TensorTest, DISABLED_UninitializedTensor) {
    Tensor x;
    EXPECT_TRUE(x.empty());
}

TEST(TensorTest, DISABLED_BasicOps) {
    Tensor a(std::vector<float>({1, 2, 3, 4}));
    Tensor b(std::vector<float>({1, 2, 3, 4}));
    a.reshape({1, 2, 2});
    b.reshape({2, 1, 2});

    EXPECT_THROW(a + b, std::exception);
}

TEST(TensorTest, DISABLED_InitializeAndSetData) {
    Tensor t;
    t.reshape({2, 3});

    EXPECT_EQ(t.size(), 6);
}

#if 0 // cant be used now
TEST(TensorTest, DISABLED_GradientOperations) {
    Tensor t;
    t.reshape({2, 3});

    EXPECT_EQ(t.grad.size(), 6);

    for (size_t i = 0; i < t.grad.size(); ++i) {
        t.grad[i] = static_cast<float>(i);
    }

    t.zero_grad();

    for (size_t i = 0; i < t.grad.size(); ++i) {
        EXPECT_FLOAT_EQ(t.grad[i], 0.0f);
    }
}
#endif

TEST(LayerTest, DISABLED_ReLU) {
    ReLU relu;
    EXPECT_EQ(relu.parameters().size(), 0);

    Tensor input;
    input.reshape({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) - 2.0f;
    }

    Tensor output;
    relu.forward(input, output);

    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_FLOAT_EQ(output[i], std::max(0.0f, input[i]));
    }

    // for (size_t i = 0; i < output.size(); ++i) {
    //     output[i] = 1.0f;
    // }

    output.backward();

    auto grad_ = output.get_grad();

    #if 0
    for (size_t i = 0; i < input.size(); ++i) {  // ???
        EXPECT_FLOAT_EQ(grad_[i], input[i] >= 0 ? 1.0f : 0.0f);
    }
    #endif

    auto input_grad = input.get_grad();
    for(size_t i = 0; i < input.size(); ++i) {
        EXPECT_FLOAT_EQ(input_grad[i], input[i] >= 0 ? 1.0f : 0.0f);
    }
}

TEST(LayerTest, DISABLED_Linear) {
    Linear linear(3, 2);

    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2);

    EXPECT_EQ(linear.weight.shape().size(), 2);
    EXPECT_EQ(linear.weight.shape()[0], 3);
    EXPECT_EQ(linear.weight.shape()[1], 2);
    EXPECT_EQ(linear.bias.shape().size(), 1);
    EXPECT_EQ(linear.bias.shape()[0], 2);

    Tensor input(true);
    input.reshape({2, 3});
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) / input.size();
    }

    Tensor output(true);
    linear.forward(input, output);

    auto shape_ = output.shape();

    EXPECT_EQ(shape_.size(), 2);
    EXPECT_EQ(shape_[0], 2);
    EXPECT_EQ(shape_[1], 2); // w.shape = [3, 2], input.shape = [2, 3] => (input @ weight).shape = [2, 2]

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] = 1.0f;
    }

    input.zero_grad();

    // linear.backward(output, input);
    output.backward();

    bool has_nonzero = false;
    auto inp_grad = input.get_grad();
    for (size_t i = 0; i < input.size(); ++i) {
        if(std::abs(inp_grad[i]) < 1e-5) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    has_nonzero = false;
    auto w_grad = linear.weight.get_grad();
    for (size_t i = 0; i < linear.weight.size(); ++i) {
        if (std::fabs(w_grad[i]) > 1e-8) {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);

    has_nonzero = false;
    auto bias_grad = linear.bias.get_grad();
    for (size_t i = 0; i < bias_grad.size(); ++i) {
        if (std::fabs(bias_grad[i]) > 1e-8) {
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
    linear.weight.init({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    linear.weight.reshape({3, 2});
    linear.bias.init({0.1f, 0.2f});
    linear.bias.reshape({2});

    Tensor input;
    input.init({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f});
    input.reshape({2, 3});

    // Forward pass
    Tensor output;
    linear.forward(input, output);

    // Check output values match PyTorch
    ASSERT_EQ(output.shape().size(), 2);
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 2);
    EXPECT_NEAR(output[0], 0.32f, 1e-5f);
    EXPECT_NEAR(output[1], 0.48f, 1e-5f);
    EXPECT_NEAR(output[2], 0.59f, 1e-5f);
    EXPECT_NEAR(output[3], 0.84f, 1e-5f);

    // Backward pass
    // output.grad({1.0f, 1.0f, 1.0f, 1.0f}); // implemented internally

    // linear.backward(output, input);
    output.backward();

    // Check input gradients match PyTorch
    auto input_grad = input.get_grad();
    ASSERT_EQ(input_grad.size(), 6);
    EXPECT_NEAR(input_grad[0], 0.3f, 1e-5f);
    EXPECT_NEAR(input_grad[1], 0.7f, 1e-5f);
    EXPECT_NEAR(input_grad[2], 1.1f, 1e-5f);
    EXPECT_NEAR(input_grad[3], 0.3f, 1e-5f);
    EXPECT_NEAR(input_grad[4], 0.7f, 1e-5f);
    EXPECT_NEAR(input_grad[5], 1.1f, 1e-5f);

    // Check weight gradients match PyTorch
    auto w_grad = linear.weight.get_grad();
    ASSERT_EQ(w_grad.size(), 6);
    EXPECT_NEAR(w_grad[0], 0.5f, 1e-5f);
    EXPECT_NEAR(w_grad[1], 0.5f, 1e-5f);
    EXPECT_NEAR(w_grad[2], 0.7f, 1e-5f);
    EXPECT_NEAR(w_grad[3], 0.7f, 1e-5f);
    EXPECT_NEAR(w_grad[4], 0.9f, 1e-5f);
    EXPECT_NEAR(w_grad[5], 0.9f, 1e-5f);

    // Check bias gradients match PyTorch
    auto bias_grad = linear.bias.get_grad();
    ASSERT_EQ(bias_grad.size(), 2);
    EXPECT_NEAR(bias_grad[0], 2.0f, 1e-5f);
    EXPECT_NEAR(bias_grad[1], 2.0f, 1e-5f);
}

TEST(ModelTest, DISABLED_ForwardAndBackwardVSTorch) {
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
    layer1->weight = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    layer1->bias = {0.1f, 0.2f};

    Linear* layer2 = static_cast<Linear*>(model.layers[2]);
    layer2->weight = {0.7f, 0.8f};
    layer2->bias = {0.3f};

    // Prepare input
    Tensor input;
    input.reshape({2, 3});
    input = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};

    // Forward pass
    Tensor output;
    model.forward(input, output);

    // Verify final output shape and values
    ASSERT_EQ(output.shape().size(), 2);
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 1);
    EXPECT_NEAR(output[0], 0.9080f, 1e-4f);
    EXPECT_NEAR(output[1], 1.3850f, 1e-4f);

    // Backward pass
    // output.grad = {1.0f, 1.0f};

    model.backward(output, input);

    // Check input gradients
    auto input_grad = input.get_grad();
    ASSERT_EQ(input_grad.size(), 6);
    EXPECT_NEAR(input_grad[0], 0.2300f, 1e-4f);
    EXPECT_NEAR(input_grad[1], 0.5300f, 1e-4f);
    EXPECT_NEAR(input_grad[2], 0.8300f, 1e-4f);
    EXPECT_NEAR(input_grad[3], 0.2300f, 1e-4f);
    EXPECT_NEAR(input_grad[4], 0.5300f, 1e-4f);
    EXPECT_NEAR(input_grad[5], 0.8300f, 1e-4f);

    // Check layer1 weight gradients
    auto l1_w_grad = layer1->weight.get_grad();
    ASSERT_EQ(l1_w_grad.size(), 6);
    EXPECT_NEAR(l1_w_grad[0], 0.3500f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[1], 0.4000f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[2], 0.4900f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[3], 0.5600f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[4], 0.6300f, 1e-4f);
    EXPECT_NEAR(l1_w_grad[5], 0.7200f, 1e-4f);

    // Check layer1 bias gradients
    auto l1_bias_grad = layer1->bias.get_grad();
    ASSERT_EQ(l1_bias_grad.size(), 2);
    EXPECT_NEAR(l1_bias_grad[0], 1.4000f, 1e-4f);
    EXPECT_NEAR(l1_bias_grad[1], 1.6000f, 1e-4f);

    // Check layer2 weight gradients
    auto l2_w_grad = layer2->weight.get_grad();
    ASSERT_EQ(l2_w_grad.size(), 2);
    EXPECT_NEAR(l2_w_grad[0], 0.9100f, 1e-4f);
    EXPECT_NEAR(l2_w_grad[1], 1.3200f, 1e-4f);

    // Check layer2 bias gradients
    auto l2_bias_grad = layer2->bias.get_grad();
    ASSERT_EQ(l2_bias_grad.size(), 1);
    EXPECT_NEAR(l2_bias_grad[0], 2.0000f, 1e-4f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
