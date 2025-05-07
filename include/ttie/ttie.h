#ifndef TTIE_H
#define TTIE_H

#include <cassert>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace ttie {

    struct GradFn;
    class Tensor;

    struct TensorImpl final {
        std::vector<size_t> shape_;
        std::vector<float> data;
        std::vector<float> grad;
        std::vector<size_t> strides;
        std::shared_ptr<GradFn> grad_fn;
        bool requires_grad = false;
    };

    struct GradFn {
        GradFn(const std::initializer_list<std::shared_ptr<TensorImpl>> lst_) : inputs(lst_) {}
        std::vector<std::shared_ptr<TensorImpl>> inputs;
        virtual void backward(TensorImpl& output) = 0; // TensorImpl -> Tensor?
    };  // GradFn не нужно знать свой истинный тип если backward написан в общем виде?

    class AddOp final : public GradFn {
        public:
            AddOp(const std::shared_ptr<TensorImpl>& a, const std::shared_ptr<TensorImpl>& b) : GradFn({a, b}) {}

            void backward(TensorImpl& output) override { // c = a + b, ∂c/∂a = b, ∂c/∂b = a;
                inputs[0]->grad = inputs[1]->data;
                inputs[1]->grad = inputs[0]->data;
            }
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
            std::shared_ptr<TensorImpl> impl_;

        public:
            Tensor() : impl_(nullptr) {}

            Tensor(const std::vector<float>& data_, bool requires_grad=false) : impl_(new TensorImpl) {
                impl_->data = data_;
                impl_->grad.resize(impl_->data.size());
                impl_->strides.resize(impl_->data.size());
                impl_->shape_ = {data_.size()};
            }

        private:

            void add_node(const GradFn& node) {}

        public:

            bool empty() const noexcept { return impl_->data.empty(); }

            void backward() { impl_->grad_fn->backward(*impl_); }

            void reshape(std::initializer_list<size_t> new_shape) {
                size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
                if(size() != new_size) {
                    throw std::invalid_argument("Invalid shape for tensor of size " + std::to_string(size()));
                }

                impl_->shape_ = new_shape;

                for(int i = 0; i < impl_->shape_.size(); ++i) { // TODO: rewrite through STL & fix
                    impl_->strides[i] = std::accumulate(std::next(impl_->shape_.begin()), impl_->shape_.end(), 1, std::multiplies<size_t>());
                }
            }

            const size_t dim(int i) const noexcept { return impl_->shape_[i]; }

            const std::vector<size_t>& shape() const noexcept { return impl_->shape_; }

            void reshape(size_t new_shape) { reshape({new_shape}); }

            float& operator[](int i) { return impl_->data[i]; }
            const float& operator[](int i) const { return impl_->data[i]; }

            float& operator[](const std::initializer_list<size_t>& idx) {
                return impl_->data[compute_index(idx)];
            }
            const float& operator[](const std::initializer_list<size_t>& idx) const {
                return impl_->data[compute_index(idx)];
            }

            Tensor& operator+=(const Tensor& lhs) {
                if(size() != lhs.size() || !std::equal(impl_->shape_.begin(), impl_->shape_.end(), lhs.impl_->shape_.begin())) {
                    throw std::invalid_argument("Can't apply operator+=() for tensors of shape " + std::to_string(size()) + " and " + std::to_string(lhs.size()));
                }

                std::transform(impl_->data.begin(), impl_->data.end(), lhs.impl_->data.begin(), impl_->data.begin(), std::plus<int>());
                impl_->grad_fn = std::make_shared<AddOp>(impl_, lhs.impl_);
                return *this;
            }

            size_t compute_index(const std::initializer_list<size_t>& idx) const {
                if(std::equal(impl_->shape_.begin(), impl_->shape_.end(), idx.begin(),
                    [](size_t a, size_t b){ return a < b; }
                )) {
                    throw std::invalid_argument("Invalid indexes for tensor of shape" + vector_to_string(impl_->shape_));
                }

                return std::inner_product(impl_->strides.begin(), impl_->strides.end(), idx.begin(), 0);
            }

            size_t size() const {
                if(impl_->data.empty()) { return 0; }

                return std::accumulate(impl_->shape_.begin(), impl_->shape_.end(), 1, std::multiplies<size_t>());
            }

            void zero_grad() {
                std::fill(impl_->grad.begin(), impl_->grad.end(), 0.0f);
            }

            const std::vector<float>& get_data() const noexcept { return impl_->data; }

            const std::vector<float>& get_grad() const noexcept { return impl_->grad; }

        friend std::ostream &operator<<(std::ostream &os, const Tensor &t) {
            os << "Tensor@" << &t;

            if(t.empty()) {
                os << "(not initialized)";
                return os;
            }

            os << "(shape=" << vector_to_string(t.shape());

            if(!t.empty()) {
                os << ", data=" << vector_to_string(t.get_data());
            } else {
                os << ", data=[no data]";
            }

            if(!t.empty()) {
                os << ", grad=" << vector_to_string(t.get_grad());
            }

            os << ")";
            return os;
        }
    };

    Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
        Tensor tmp{lhs};
        tmp += rhs;
        return tmp;
    }

    struct Layer {
        virtual void forward(const Tensor &input, Tensor &output) = 0;
        virtual void backward(const Tensor &grad_output, Tensor &grad_input) = 0;
        virtual std::string to_string() const = 0;
        virtual std::vector<Tensor *> parameters() = 0;
        virtual ~Layer() {}
    };

    #if 0
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
    #endif

}

#endif // TTIE_H
