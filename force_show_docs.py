"""
强制显示文档的辅助模块
"""

# 全局标志，控制是否显示文档
FORCE_SHOW_DOCS = True

def set_force_show_docs(value=True):
    """设置是否强制显示文档的全局标志"""
    global FORCE_SHOW_DOCS
    FORCE_SHOW_DOCS = value
    
def should_show_docs(original_value=False):
    """根据全局标志决定是否显示文档"""
    global FORCE_SHOW_DOCS
    # 始终返回 True
    return True 