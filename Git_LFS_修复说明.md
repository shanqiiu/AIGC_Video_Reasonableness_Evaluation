# Git LFS 锁定 API 问题修复说明

## 问题描述

在执行 `git push -u origin main` 时出现以下错误：

```
Remote "origin" does not support the Git LFS locking API. Consider disabling it with:
```

## 原因分析

这个错误是因为远程仓库（GitHub）不支持 Git LFS 的锁定 API，但你的本地 Git LFS 配置启用了锁定验证功能。

## 解决方案

已执行以下命令禁用 LFS 锁定验证：

```bash
git config lfs.locksverify false
```

## 验证

可以通过以下命令验证配置：

```bash
git config --get lfs.locksverify
```

应该返回：`false`

## 注意事项

1. **此配置的作用域**：
   - 当前设置只影响当前仓库
   - 如果需要全局设置，可以使用：`git config --global lfs.locksverify false`

2. **LFS 锁定验证的作用**：
   - LFS 锁定验证用于防止多个用户同时修改同一个大文件
   - 对于大多数个人项目和小团队，禁用此功能不会造成问题
   - GitHub 的免费 LFS 不支持锁定 API，所以需要禁用

3. **其他可能的解决方案**：
   - 如果只想为当前仓库禁用，使用：`git config lfs.locksverify false`
   - 如果希望全局禁用，使用：`git config --global lfs.locksverify false`
   - 如果使用 GitLab 或其他支持 LFS 锁定的服务，可以保持启用

## 现在可以正常推送了

现在你可以再次尝试：

```bash
git push -u origin main
```

应该不会再出现 LFS 锁定 API 的警告了。

