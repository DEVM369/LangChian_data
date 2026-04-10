
# 1. **LangChain介绍**

## 1.1. **LangChain的由来**

LangChain 是一个**专门用于开发大语言模型（LLM）应用**的开源框架。它不仅仅是对大模型 API 的简单调用，而是为开发者提供了一套完整的“脚手架”（包含预构建的 Agent 智能体架构和丰富的工具集），帮助开发者无需从零搭建，即可快速构建复杂的企业级 AI 应用。

LangChain这一开源框架诞生于2022年10月，由哈佛大学的Harrison Chase（哈里森·蔡斯）创建， **其名称来源于"Language"（语言模型）和"Chain"（链式连接）的组合** ，体现了其核心设计理念—— **将孤立的大模型与外部的数据源、计算工具像“链条”一样无缝串联，从而大幅拓展 AI 的能力边界** 。

LangChain的诞生源于一个关键洞察：单一的大模型在实际生产环境中存在明显局限，LangChain 通过引入外部模块完美解决了这些问题：

* **知识滞后** ：大模型无法获取训练截止日期之后的信息。**（解法：连接外部数据源与搜索引擎，实时检索信息）**；
* **缺乏行动力** ：大模型只能生成文本，无法与外部系统交互。**（解法：提供工具集成，允许模型调用 API、读写数据库或执行代码）**；
* **无上下文记忆** ：大模型本身是无状态的，无法维持连贯的多轮交互。**（解法：引入记忆机制 Memory，保持对话状态）**。

所以要构建真正实用的AI应用，必须将大语言模型与外部工具、数据源和记忆机制有机结合，从而催生了LangChain框架的设计理念。

2024年是LangChain架构重大变革的一年。随着开发者从构建原型转向生产环境部署，对更精细工作流控制的需求日益增长，LangChain团队**推出了LangGraph**作为底层智能体编排框架，并将原有的链和智能体标记为弃用，转而采用基于LangGraph构建的统一智能体抽象。2025年10月20日LangChain团队正式发布LangChain v1.0.0与LangGraph v1.0.0，标志着框架的成熟和标准化，为企业级AI应用提供了稳定基础。

LangChain&LangGraph文档地址如下：

英文文档地址：[https://docs.langchain.com/oss/python/langchain/overview](https://docs.langchain.com/oss/python/langchain/overview)

中文文档地址：[https://docs.langchain.org.cn/oss/python/langchain/overview](https://docs.langchain.org.cn/oss/python/langchain/overview)

## 1.2. **LangChain核心特点**

LangChain作为一个成熟的大模型应用开发框架，具备四大核心特点：

**1. 统一模型接口 (Model I/O)**

* **核心定义** ：提供标准化的 API 抽象层，屏蔽底层模型的差异。
* **解决痛点** ：原生调用不同大模型（如 OpenAI、Google 等）时，接口和参数格式各异，导致代码耦合度高、切换成本大。
* **实现方式** ：开发者只需编写一套标准代码，即可无缝切换不同的底层模型。这不仅消除了“厂商锁定（Vendor Lock-in）”的风险，还能让项目快速迭代最新模型。

**2. 模块化架构与 LCEL (Modular Architecture)**

* **核心定义** ：将大模型应用拆解为高度解耦、可复用的组件（提示词、模型、输出解析器等）。
* **解决痛点** ：避免了将复杂的 AI 逻辑写成一团乱麻（意大利面条式代码），提升了开发效率和可维护性。
* **实现方式** ：引入了  **LangChain 表达式语言（LCEL）** 。开发者可以通过管道符（`|`）将组件拼接成声明式的工作流（例如：`检索器 | 提示模板 | LLM`）。这种架构天然支持并行执行、异步调用和流式输出。

**3. 智能体与工具调用 (Agents & Tools)**

* **核心定义** ：赋予大模型自主决策和执行物理操作的能力。
* **解决痛点** ：纯文本模型只能“纸上谈兵”，缺乏获取实时数据和操作外部系统的能力。
* **实现方式** ：
  * **大脑（Agent）** ：基于“观察-思考-行动（ReAct）”的推理框架，让大模型自主规划任务步骤。
  * **手脚（Tools）** ：封装了标准化接口的外部函数（如搜索引擎、数据库查询、API 调用）。智能体会自动选择并调用合适的工具，直到彻底解决用户问题。

**4. 记忆管理机制 (Memory Management)**

* **核心定义** ：为应用提供管理对话上下文的状态维持能力。
* **解决痛点** ：底层大语言模型本质上是“无状态（Stateless）”的，无法天然记住多轮对话的历史信息。
* **实现方式** ：采用分层的记忆存储架构。支持将会话数据存储在内存、文件或数据库中；并允许开发者灵活配置记忆策略（如保留全局历史的“长期记忆”，或仅保留最近几轮对话的“短期记忆”），从而实现连贯的个性化交互。

## 1.3. **LangChain使用场景**

LangChain广泛适用于如下场景：

**1. 检索增强生成 (RAG - Retrieval-Augmented Generation)**

* **核心定义** ：为大模型外接专属的外部知识库（如文档、数据库），先检索后生成。
* **解决痛点** ：消除大模型的知识滞后性，并大幅减少“幻觉（瞎编）”现象。
* **典型场景** ：企业内部知识库问答、基于私有文档的智能客服。

**2. 智能体构建 (Agent)**

* **核心定义** ：将大模型作为“推理引擎”，使其能够自主规划步骤并调用外部工具。
* **解决痛点** ：打破模型只能“生成文本”的局限，使其具备执行复杂现实任务的行动力。
* **典型场景** ：全自动市场调研（自动搜索、抓取数据并写出简报）、自动化旅行规划（自动调用航班和酒店 API）。

**3. 智能对话系统 (Chatbot)**

* **核心定义** ：结合记忆（Memory）机制，构建具备上下文感知能力的多轮对话机器人。
* **解决痛点** ：传统大模型 API 缺乏状态保持，无法记住用户的历史交互和个性化偏好。
* **典型场景** ：具备深层记忆的电商售后客服、长效陪伴的教育辅导助手。

**4. 数据连接与处理 (Data Connection & Processing)**

* **核心定义** ：利用大模型的语言理解能力，与各种结构化或非结构化数据进行交互。
* **解决痛点** ：降低数据分析门槛，免去手动编写复杂查询代码或人工提取信息的繁琐。
* **典型场景** ：Text-to-SQL（用自然语言直接查询关系型数据库）、从长篇 PDF 合同中精准抽取金额与条款信息。

**5. 结构化内容生成 (Content Generation)**

* **核心定义** ：通过提示词模板（Prompt Templates）和输出解析器，控制大模型生成符合特定规范的内容。
* **解决痛点** ：大模型自由发散度高，难以直接产出格式严格、符合业务质量要求的标准化文档。
* **典型场景** ：基于业务数据自动化生成周报、基于固定模板自动起草规范的法律文书。

**6. 多模态应用开发 (Multimodal Applications)**

* **核心定义** ：拓展单一的文本处理能力，接入并处理图像、音频、视频等媒体数据。
* **解决痛点** ：满足真实业务场景中非文本类数据的交互与分析需求。
* **典型场景** ：用户上传图片后系统自动分析并回答相关问题、语音实时转换与交互处理系统。

## 1.4. **LangChain快速上手**

### **1.4.1. python环境准备**

LangChain 1.2版本要求Python版本为3.10+以上，这里我们使用anconda 来创建和管理python环境，这里默认大家已经安装好了anconda(也可以安装minicoda)，本课程使用python环境为3.13.11版本，按照如下方式创建python3.13环境：

```
#创建一个名为langchain_v1.2的环境，指定Python版本是3.13.11
conda create --name langchain_v1.2 python=3.13.11

#查看anconda安装好的python环境
conda env list
langchain_v1.2           D:\ProgramData\miniconda3\envs\langchain_v1.2

#在命令行窗口切换到某python环境
conda activate langchain_v1.2
(langchain_v1.2) C:\Users\wubai>python -V
Python 3.13.11

#在命令行退出当前python环境
conda deactivate langchain_v1.2

#删除一个已有的anconda管理的python环境
conda remove --name langchain_v1.2 --all
```

### **1.4.2. 创建项目及配置**

创建好Python环境后，创建python项目并指定python环境，同时把基本的配置文件配置好。按照如下步骤进行设置。

**1) 创建python项目**

在IDEA中创建Langchainv12Project项目并指定python环境为“langchain_v1.2”。

![image.png](https://fynotefile.oss-cn-zhangjiakou.aliyuncs.com/fynote/fyfile/70313/1774875116001/134a81f1cea54af39ec9d4cd46f258bd.png)

**2) 安装必要依赖**

想要正常使用LangChain还需要在该环境中安装如下依赖。

```
# 切换conda环境
conda activate langchain_v1.2

#安装依赖
python -m pip install langchain==1.2.0  langchain-deepseek==1.0.1  dotenv==0.9.9 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

注意：langchain-deepseek 是使用deepseek 大模型必要依赖，dotenv是从项目根目录.env文件中加载自定义环境变量必要依赖。

**3) 创建.env文件**

在项目根目录下创建“.env”文件并写入如下内容，该文件中后续配置一些大模型的API_KEY和BASE_URL。

```
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

**4) 创建“env_** [**utils.py**](http://utils.py/)**”文件**

该文件中通过dotenv加载并获取.env文件中配置的环境变量，后续方便在项目中使用这些环境变量配置的值。

```
import os

from dotenv import load_dotenv

# override=True 确保.env文件优先
load_dotenv(override=True)

# 从环境变量读取配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
```

**5) 创建“my_** [**llm.py**](http://llm.py/)**”文件**

该文件后续会创建各种大模型，方便在项目中使用。如下，创建deepseek LLM。

```
from langchain_deepseek import ChatDeepSeek

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# 创建DeepSeek LLM
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    api_base=DEEPSEEK_BASE_URL,
    model="deepseek-chat",
)
```

### **1.4.3. 快速上手案例-Agent查询天气**

该案例中使用langchain创建agent，该agent可以调用查询天气工具进行天气查询，天气查询工具通过代码模拟生成。

quick_[start.py](http://start.py/) 代码如下：

```
from langchain.agents import create_agent

from my_llm import deepseek_llm


def get_weather(city: str) -> str:
    # 模拟天气查询
    """获取给定城市的天气。"""
    return f"{city} 天气晴朗！"

# 创建Agent
agent = create_agent(
    model=deepseek_llm,
    tools=[get_weather],
    system_prompt="你是一个助手，你可以查询城市的天气。",
)

# 调用Agent
resp = agent.invoke(
    {"messages": [{"role": "user", "content": "查询北京的天气"}]}
)

print(resp)
```

以上代码运行结果如下：

```
{'messages': [
	HumanMessage(
		content='查询北京的天气', 
		additional_kwargs={}, 
		response_metadata={}, 
		id='ead4d302-a74e-4bab-9a3e-35a6233ed4c7'
		), 

	AIMessage(
		content='我来帮您查询北京的天气。', 
		additional_kwargs={'refusal': None},
		response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 309, 'total_tokens': 359, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 256}, 'prompt_cache_hit_tokens': 256, 'prompt_cache_miss_tokens': 53}, 'model_provider': 'deepseek', 'model': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '222ab7cd-7984-479f-ae03-659aff0355b4', 'finish_reason': 'tool_calls', 'logprobs': None}, 
		id='lc_run--019b9785-13e6-7d52-b947-41bf1a1e33dd-0', 
		tool_calls=[{'name': 'get_weather', 'args': {'city': '北京'}, 'id': 'call_00_Y1R8gZcaaQ4IXPqKfUUBns4G', 'type': 'tool_call'}], 
		invalid_tool_calls=[], 
		usage_metadata={'input_tokens': 309, 'output_tokens': 50, 'total_tokens': 359, 'input_token_details': {'cache_read': 256}, 'output_token_details': {}}
		), 

	ToolMessage(
		content='北京 天气晴朗！', 
		name='get_weather', 
		id='6f3a21fd-cb05-4b3b-9fca-f1584d78ad01', 
		tool_call_id='call_00_Y1R8gZcaaQ4IXPqKfUUBns4G'
		), 

	AIMessage(
		content='根据查询结果，北京今天的天气是晴朗的！', 
		additional_kwargs={'refusal': None}, 
		response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 381, 'total_tokens': 392, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 320}, 'prompt_cache_hit_tokens': 320, 'prompt_cache_miss_tokens': 61}, 'model_provider': 'deepseek', 'model': 'deepseek-chat', 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '9c667481-e7af-4169-88ad-2b71c530a395', 'finish_reason': 'stop', 'logprobs': None}, 
		id='lc_run--019b9785-1f25-7881-abb9-656577b5ab60-0', 
		tool_calls=[], 
		invalid_tool_calls=[], 
		usage_metadata={'input_tokens': 381, 'output_tokens': 11, 'total_tokens': 392, 'input_token_details': {'cache_read': 320}, 'output_token_details': {}})
]}
```
