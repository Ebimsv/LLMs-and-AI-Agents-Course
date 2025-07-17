# LLMs and Agents in Production 🚀

A comprehensive 6-day tutorial series for developers interested in **Generative AI**, **Large Language Models**, and deploying **AI agents** in real-world scenarios.

## 📋 Overview

This repository contains hands-on tutorials that guide you through the complete journey of working with Large Language Models (LLMs) - from understanding core concepts to building production-ready applications. Each day builds upon the previous, creating a structured learning path for developers who want to leverage LLMs in their projects.

### 🎯 Target Audience
- **Developers** interested in practical applications of Generative AI
- **Software Engineers** looking to integrate LLMs into their applications
- **Data Scientists** wanting to explore LLM capabilities
- **Anyone** with basic Python knowledge eager to learn about modern AI

## 🛠️ Prerequisites

### Required Skills
- Basic Python programming knowledge
- Familiarity with command line operations
- Understanding of HTTP APIs (helpful but not required)

### System Requirements
- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **Memory**: At least 8GB RAM (16GB recommended for local LLMs)
- **Storage**: 10GB+ free space for model downloads

### Software Dependencies
- Python 3.8+
- pip (Python package manager)
- Git (for cloning this repository)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ebimsv/llms-agents-production.git
   cd llms-agents-production
   ```

2. **Set up your environment**
   ```bash
   # Create a virtual environment
   python -m venv llms_env
   
   # Activate the environment
   # On Windows:
   llms_env\Scripts\activate
   # On macOS/Linux:
   source llms_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Follow the tutorials**
   - Start with [Day 1: Understanding LLMs](./day-1-understanding-llms/)
   - Progress through each day sequentially
   - Each day includes practical exercises and code examples

## 📚 Content Overview

### Day 1: What is an LLM? Understanding the Core Concepts
**📁 Location**: `./day-1-understanding-llms/`

- **Core Concepts**: What are Large Language Models?
- **Architecture**: Understanding transformer models
- **Capabilities**: What LLMs can and cannot do
- **Real-world Applications**: Use cases in production
- **Hands-on**: Setting up your first LLM interaction

**Key Takeaways**: Foundation knowledge of LLM technology and capabilities

---

### Day 2: Intro to LLM APIs with Python: OpenAI, Ollama, and Beyond
**📁 Location**: `./day-2-llm-apis/`

- **API Fundamentals**: Understanding REST APIs for LLMs
- **OpenAI Integration**: Using GPT models via API
- **Ollama Setup**: Running local LLMs
- **Error Handling**: Best practices for production
- **Cost Optimization**: Managing API usage efficiently

**Key Takeaways**: Practical experience with multiple LLM providers and APIs

---

### Day 3: What is Ollama? Running LLMs Locally with Just One Command
**📁 Location**: `./day-3-ollama-local-llms/`

- **Ollama Installation**: Setting up local LLM environment
- **Model Management**: Downloading and managing different models
- **Performance Optimization**: Hardware considerations
- **Streamlit Integration**: Building web interfaces
- **Deployment Strategies**: Local vs cloud considerations

**Key Takeaways**: Complete local LLM setup and web application development

---

### Day 4: Exploring LLM Leaderboards: How to Choose the Right Model
**📁 Location**: `./day-4-model-selection/`

- **Model Evaluation**: Understanding benchmarks and metrics
- **Performance Comparison**: Speed vs accuracy trade-offs
- **Use Case Matching**: Selecting models for specific tasks
- **Cost Analysis**: Budget considerations for different models
- **Hands-on Comparison**: Testing multiple models on the same task

**Key Takeaways**: Systematic approach to model selection for production use

---

### Day 5: Prompt Engineering 101: Writing Effective Prompts
**📁 Location**: `./day-5-prompt-engineering/`

- **Prompt Fundamentals**: Structure and best practices
- **Techniques**: Few-shot learning, chain-of-thought, etc.
- **Optimization**: Iterative prompt improvement
- **Evaluation**: Measuring prompt effectiveness
- **Production Patterns**: Reusable prompt templates

**Key Takeaways**: Mastery of prompt engineering for reliable LLM outputs

---

### Day 6: Building a Website Summarizer with BeautifulSoup and LLaMA
**📁 Location**: `./day-6-website-summarizer/`

- **Web Scraping**: Using BeautifulSoup for content extraction
- **LLM Integration**: Combining web data with LLM processing
- **Error Handling**: Robust production-ready code
- **User Interface**: Building intuitive web applications
- **Deployment**: Making your application accessible

**Key Takeaways**: Complete end-to-end application development with LLMs

## 🏗️ Project Structure

```
llms-agents-production/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── setup.sh                           # Quick setup script
├── day-1-understanding-llms/          # Day 1: LLM Fundamentals
│   ├── README.md
│   ├── concepts.ipynb
│   └── exercises/
├── day-2-llm-apis/                    # Day 2: API Integration
│   ├── README.md
│   ├── openai_examples.py
│   ├── ollama_examples.py
│   └── api_comparison.ipynb
├── day-3-ollama-local-llms/           # Day 3: Local LLM Setup
│   ├── README.md
│   ├── ollama_setup.md
│   ├── streamlit_apps/
│   └── performance_guide.md
├── day-4-model-selection/              # Day 4: Model Evaluation
│   ├── README.md
│   ├── model_comparison.ipynb
│   ├── benchmarks/
│   └── selection_guide.md
├── day-5-prompt-engineering/           # Day 5: Prompt Design
│   ├── README.md
│   ├── prompt_examples.py
│   ├── techniques/
│   └── evaluation_tools/
├── day-6-website-summarizer/           # Day 6: Complete Application
│   ├── README.md
│   ├── app.py
│   ├── scraper.py
│   ├── templates/
│   └── static/
├── shared/                             # Common utilities
│   ├── utils.py
│   ├── config.py
│   └── examples/
├── docs/                               # Additional documentation
│   ├── deployment.md
│   ├── troubleshooting.md
│   └── best_practices.md
└── assets/                             # Images and resources
    ├── diagrams/
    └── screenshots/
```

## 🎯 Key Features

### ✅ **Beginner-Friendly**
- Step-by-step instructions
- Clear explanations of concepts
- Minimal prerequisites
- Troubleshooting guides

### ✅ **Production-Ready**
- Error handling best practices
- Scalability considerations
- Cost optimization strategies
- Deployment guidelines

### ✅ **Hands-On Learning**
- Interactive Jupyter notebooks
- Real-world code examples
- Practical exercises
- Complete applications

### ✅ **Comprehensive Coverage**
- Multiple LLM providers
- Various use cases
- Performance optimization
- Modern development practices

## 🛠️ How to Use This Repository

### For Beginners
1. Start with Day 1 and progress sequentially
2. Complete all exercises in each day
3. Experiment with the provided code examples
4. Use the troubleshooting guides if you encounter issues

### For Experienced Developers
1. Skip to relevant days based on your needs
2. Focus on production-ready code examples
3. Explore advanced topics in each day's `advanced/` folder
4. Contribute improvements and additional examples

### For Educators
1. Use as a structured curriculum for AI/ML courses
2. Adapt exercises for different skill levels
3. Leverage the modular structure for custom learning paths

## 📖 Additional Resources

### Documentation
- [Deployment Guide](./docs/deployment.md) - Production deployment strategies
- [Troubleshooting](./docs/troubleshooting.md) - Common issues and solutions
- [Best Practices](./docs/best_practices.md) - Industry standards and recommendations

### External Links
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Ollama Documentation](https://ollama.ai/docs)
- [Hugging Face Models](https://huggingface.co/models)
- [Streamlit Documentation](https://docs.streamlit.io/)

### Community
- [GitHub Issues](https://github.com/yourusername/llms-agents-production/issues) - Report bugs or request features
- [Discussions](https://github.com/yourusername/llms-agents-production/discussions) - Ask questions and share experiences
- [Medium Series](https://medium.com/@yourusername) - Original tutorial articles

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](./CONTRIBUTING.md) for details on:
- Code style and standards
- Pull request process
- Adding new tutorials or examples
- Improving documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- Ollama team for local LLM capabilities
- Streamlit for web application framework
- The open-source community for various tools and libraries

---

**Ready to start your LLM journey?** 🚀

Begin with [Day 1: Understanding LLMs](./day-1-understanding-llms/) and transform your development skills with the power of Large Language Models!

---

*Last updated: July 2025* 