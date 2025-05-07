#ifndef TTIE_H
#define TTIE_H

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ttie {

    enum Node_t {
        BinOp
    };

    class Node {
        Node* parent_;
        Node_t type_;
    };

    class BinOp : public Node {
        Node* left_, right_;
    };

    template <typename T>
    static std::string vector_to_string(const std::vector<T> &vec, size_t limit = 5) {
        std::stringstream ss;
        ss << "[";
        size_t preview_size = std::min(limit, vec.size());
        for (size_t i = 0; i < preview_size; ++i) {
            ss << vec[i];
            if (i < preview_size - 1) {
                ss << ", ";
            }
        }
        if (vec.size() > preview_size) {
            ss << ", ...";
        }
        ss << "]";
        return ss.str();
    }

    class Tensor final {
        private:
            std::vector<size_t> shape_;
            std::vector<float> data;
            std::vector<float> grad;
            std::vector<size_t> strides;
            Node* grad_fn;

        public:
            Tensor() : grad_fn(nullptr) {}

            Tensor(const std::vector<float>& data_) : data(data_), shape_({data_.size()}), grad(data_.size()), strides(data_.size(), 1) {}

            /* big five */

            bool empty() const noexcept { return data.empty(); }

            void reshape(std::initializer_list<size_t> new_shape) {
                size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
                if(size() != new_size) {
                    throw std::invalid_argument("Invalid shape for tensor of size " + std::to_string(new_size));
                }

                shape_ = new_shape;

                for(int i = 0; i < shape_.size(); ++i) { // rewrite through STL
                    strides[i] = std::accumulate(std::next(shape_.begin()), shape_.end(), 1, std::multiplies<size_t>());
                }
            }

            const size_t dim(int i) const noexcept { return shape_[i]; }

            const std::vector<size_t>& shape() const noexcept { return shape_; }

            void reshape(size_t new_shape) { reshape({new_shape}); }

            float& operator[](int i) { return data[i]; }
            const float& operator[](int i) const { return data[i]; }

            float& operator[](const std::initializer_list<size_t>& idx) {
                return data[compute_index(idx)];
            }
            const float& operator[](const std::initializer_list<size_t>& idx) const {
                return data[compute_index(idx)];
            }

            size_t compute_index(const std::initializer_list<size_t>& idx) const {
                if(std::equal(shape_.begin(), shape_.end(), idx.begin(),
                    [](size_t a, size_t b){ return a < b; }
                )) {
                    throw std::invalid_argument("Invalid indexes for tensor of shape" + vector_to_string(shape_));
                }

                return std::inner_product(strides.begin(), strides.end(), idx.begin(), 0);
            }

            #if 0
            bool validate_shape() const {
                if(shape.empty()) {
                    return false;
                }

                return std::if_any(shape.begin(), shape.end(),
                    [](auto it){ return it == 0; }
                );
            }
            #endif

            size_t size() const {       // переписать на class, убрать проверку при каждом вызове
                #if 0
                if(!validate_shape()) {
                    throw std::invalid_argument("Invalid tensor shape");
                }
                #endif

                return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>());
            }

            void zero_grad() {
                std::fill(grad.begin(), grad.end(), 0.0f);
            }

        friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
            os << "Tensor@" << &t;

            if(t.empty()) {
                os << "(not initialized)";
                return os;
            }

            os << "(shape=" << vector_to_string(t.shape());

            if(!t.data.empty()) {
                os << ", data=" << vector_to_string(t.data);
            } else {
                os << ", data=[no data]";
            }

            if(!t.grad.empty()) {
                os << ", grad=" << vector_to_string(t.grad);
            }

            os << ")";
            return os;
        }
    };

    Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
        Tensor tmp{lhs};

    }

    struct Layer {
        virtual void forward(const Tensor &input, Tensor &output) = 0;
        virtual void backward(const Tensor &grad_output, Tensor &grad_input) = 0;
        virtual std::string to_string() const = 0;
        virtual std::vector<Tensor *> parameters() = 0;
        virtual ~Layer() {}
    };

    struct Linear : Layer {
        Tensor weight;
        Tensor bias;

        Linear(size_t in_features, size_t out_features) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dis(-0.1f, 0.1f);

            weight.reshape({in_features, out_features});
            // weight.resize();

            for(size_t i = 0; i < in_features * out_features; ++i) {
                weight[i] = dis(gen);
            }

            bias.reshape(out_features);
            // bias.resize();

            for(size_t i = 0; i < out_features; ++i) {
                bias[i] = dis(gen);
            }
        }

        std::vector<Tensor *> parameters() override { return {&weight, &bias}; }

        void forward(const Tensor &input, Tensor &output) override {
            size_t in_features = weight[0];
            size_t out_features = weight[1];
            output.reshape({input.dim(0), out_features});
            // output.resize();

            for(size_t i = 0; i < input.dim(0); ++i) {
                for(size_t j = 0; j < out_features; ++j) {
                    output[i * out_features + j] = bias[j];
                    for(size_t k = 0; k < in_features; ++k) {
                        output[i * out_features + j] += input[i * in_features + k] * weight[k * out_features + j];
                    }
                }
            }
        }
        #if 0
        void backward(const Tensor &output, Tensor &input) override {
            size_t in_features = weight.dim(0);
            size_t out_features = weight.dim(1);
            size_t batch_size = output.dim(0);

            // input.resize_grad();
            // weight.resize_grad();
            // bias.resize_grad();

            for(size_t i = 0; i < batch_size; ++i) {
                for(size_t j = 0; j < in_features; ++j) {
                    input.grad[i * in_features + j] = 0;
                    for(size_t k = 0; k < out_features; ++k) {
                        input.grad[i * in_features + j] += output.grad[i * out_features + k] * weight.data[j * out_features + k];
                        weight.grad[j * out_features + k] += output.grad[i * out_features + k] * input.data[i * in_features + j];
                    }
                }
            }

            for(size_t i = 0; i < batch_size; ++i) {
                for(size_t k = 0; k < out_features; ++k) {
                    bias.grad[k] += output.grad[i * out_features + k];
                }
            }
        }
        #endif

        std::string to_string() const override {
            std::stringstream ss;
            ss << "Linear(in_features=" << weight.dim(0) << ", out_features=" << weight.dim(1) << ")";
            return ss.str();
        }
    };

    struct ReLU : Layer {
        std::vector<Tensor *> parameters() override { return {}; }

        void forward(const Tensor &input, Tensor &output) override {
            output.shape = input.shape;
            output.resize();
            for(size_t i = 0; i < input.data.size(); ++i) {
                output.data[i] = std::max(0.0f, input.data[i]);
            }
        }

        void backward(const Tensor &output, Tensor &input) override {
            input.resize_grad();
            for(size_t i = 0; i < output.data.size(); ++i) {
                input.grad[i] = (output.data[i] > 0) ? output.grad[i] : 0;
            }
        }

        std::string to_string() const override { return "ReLU()"; }
    };

    struct Model {
        std::vector<Layer *> layers;
        std::vector<Tensor> activations;

        void add_layer(Layer *layer) { layers.push_back(layer); }

        void forward(const Tensor &input, Tensor &output) {
            activations.resize(layers.size() - 1);

            const Tensor *current = &input;
            for(size_t i = 0; i < layers.size(); ++i) {
                Tensor *next = (i == layers.size() - 1) ? &output : &activations[i];
                layers[i]->forward(*current, *next);
                current = next;
            }
        }

        void backward(const Tensor &output, Tensor &input) {
            if(activations.size() != layers.size() - 1) {
                throw std::runtime_error("Forward pass must be called before backward pass");
            }

            const Tensor *current = &output;
            for(int i = layers.size() - 1; i >= 0; --i) {
                Tensor *prev = (i > 0) ? &activations[i - 1] : &input;
                layers[i]->backward(*current, *prev);
                current = prev;
            }
        }

        std::vector<Tensor *> parameters() {
            std::vector<Tensor *> params;
            for(Layer *layer : layers) {
                auto layer_params = layer->parameters();
                params.insert(params.end(), layer_params.begin(), layer_params.end());
            }
            return params;
        }

        std::string to_string() const {
            std::stringstream ss;
            for(Layer *layer : layers) {
                ss << layer->to_string() << "\n";
            }
            return ss.str();
        }

        ~Model() {
            for(Layer *layer : layers) {
                delete layer;
            }
        }
    };

    Tensor mse_loss(const Tensor &pred, const Tensor &target) {
        if(pred.data.size() != target.data.size()) {
            throw std::invalid_argument("Prediction and target tensors must have same size");
        }

        Tensor loss;
        loss.shape = {1};
        loss.resize();
        loss.data[0] = 0.0f;

        for (size_t i = 0; i < pred.data.size(); ++i) {
            float diff = pred.data[i] - target.data[i];
            loss.data[0] += diff * diff;
        }
        loss.data[0] /= pred.data.size();
        return loss;
    }

}

#endif // TTIE_H
