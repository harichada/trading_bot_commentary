import React, {useState} from 'react';
import {
  View,
  Text,
  TextInput,
  StyleSheet,
  ScrollView,
} from 'react-native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {useServerConfig} from '../context/ServerConfigContext';
import {useAuth} from '../context/AuthContext';
import {GradientCard} from '../components/GradientCard';
import {HapticButton} from '../components/HapticButton';
import {DetailRow} from '../components/DetailRow';
import {ConfirmDialog} from '../components/ConfirmDialog';
import {colors, typography, spacing, borderRadius} from '../theme';
import Toast from 'react-native-toast-message';

export function SettingsScreen() {
  const {serverUrl, apiKey, setServerUrl, setApiKey, clearConfig} =
    useServerConfig();
  const {user, logout} = useAuth();

  const [urlDraft, setUrlDraft] = useState(serverUrl);
  const [keyDraft, setKeyDraft] = useState(apiKey);
  const [saving, setSaving] = useState(false);
  const [showReset, setShowReset] = useState(false);
  const [showLogout, setShowLogout] = useState(false);

  const dirty = urlDraft !== serverUrl || keyDraft !== apiKey;

  const handleSave = async () => {
    if (!urlDraft.trim()) {
      Toast.show({type: 'error', text1: 'Server URL is required'});
      return;
    }
    if (!keyDraft.trim()) {
      Toast.show({type: 'error', text1: 'API Key is required'});
      return;
    }
    setSaving(true);
    try {
      await setServerUrl(urlDraft);
      await setApiKey(keyDraft);
      Toast.show({type: 'success', text1: 'Settings saved'});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Save failed', text2: err.message});
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    await clearConfig();
    await logout();
    setShowReset(false);
  };

  const handleLogout = async () => {
    await logout();
    setShowLogout(false);
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Animated.View entering={FadeInDown.delay(50).duration(300)}>
        <Text style={styles.title}>Settings</Text>
      </Animated.View>

      {/* Server Connection */}
      <Animated.View entering={FadeInDown.delay(100).duration(300)}>
        <GradientCard accent="cyan" style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="server-outline" size={20} color={colors.cyan} />
            <Text style={styles.sectionTitle}>Server Connection</Text>
          </View>

          <Text style={styles.label}>SERVER URL</Text>
          <View style={styles.inputWrapper}>
            <Ionicons name="globe-outline" size={18} color={colors.textMuted} />
            <TextInput
              style={styles.input}
              value={urlDraft}
              onChangeText={setUrlDraft}
              placeholder="https://your-server.example.com"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
              keyboardType="url"
            />
          </View>

          <Text style={[styles.label, {marginTop: spacing.md}]}>API KEY</Text>
          <View style={styles.inputWrapper}>
            <Ionicons name="key-outline" size={18} color={colors.textMuted} />
            <TextInput
              style={styles.input}
              value={keyDraft}
              onChangeText={setKeyDraft}
              placeholder="Enter your API key"
              placeholderTextColor={colors.textMuted}
              autoCapitalize="none"
              autoCorrect={false}
              secureTextEntry
            />
          </View>

          <HapticButton
            label={saving ? 'Saving...' : 'Save Changes'}
            icon="checkmark-circle"
            onPress={handleSave}
            disabled={!dirty || saving}
            loading={saving}
            size="md"
            style={{marginTop: spacing.lg, opacity: dirty ? 1 : 0.4}}
          />
        </GradientCard>
      </Animated.View>

      {/* Account */}
      {user && (
        <Animated.View entering={FadeInDown.delay(200).duration(300)}>
          <GradientCard accent="purple" style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="person-outline" size={20} color={colors.purple} />
              <Text style={styles.sectionTitle}>Account</Text>
            </View>
            <DetailRow label="Name" value={user.name} />
            <DetailRow label="Email" value={user.email} />
            <DetailRow label="Provider" value={user.provider} />
            <HapticButton
              label="Logout"
              icon="log-out-outline"
              variant="danger"
              onPress={() => setShowLogout(true)}
              size="md"
              style={{marginTop: spacing.md}}
            />
          </GradientCard>
        </Animated.View>
      )}

      {/* Danger Zone */}
      <Animated.View entering={FadeInDown.delay(300).duration(300)}>
        <GradientCard accent="red" style={styles.section}>
          <View style={styles.sectionHeader}>
            <Ionicons name="warning-outline" size={20} color={colors.red} />
            <Text style={styles.sectionTitle}>Danger Zone</Text>
          </View>
          <Text style={styles.dangerText}>
            Reset will clear your server URL and API key. You will need to reconfigure the app.
          </Text>
          <HapticButton
            label="Reset Server Config"
            icon="trash-outline"
            variant="danger"
            onPress={() => setShowReset(true)}
            size="md"
            style={{marginTop: spacing.md}}
          />
        </GradientCard>
      </Animated.View>

      {/* Confirm Dialogs */}
      <ConfirmDialog
        visible={showReset}
        title="Reset Server Config"
        message="This will clear your server URL and API key. You will need to re-enter them to use the app."
        confirmText="Reset"
        destructive
        onConfirm={handleReset}
        onCancel={() => setShowReset(false)}
      />
      <ConfirmDialog
        visible={showLogout}
        title="Logout"
        message="Are you sure you want to log out?"
        confirmText="Logout"
        destructive
        onConfirm={handleLogout}
        onCancel={() => setShowLogout(false)}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingTop: spacing.xl + 40, paddingBottom: spacing.xxxl},
  title: {...typography.h1, color: colors.textPrimary, marginBottom: spacing.lg},
  section: {marginBottom: spacing.md},
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    marginBottom: spacing.md,
  },
  sectionTitle: {...typography.h3, color: colors.textPrimary},
  label: {
    ...typography.label,
    color: colors.textMuted,
    marginBottom: spacing.xs,
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
    backgroundColor: colors.bgInput,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: borderRadius.lg,
    paddingHorizontal: spacing.md,
  },
  input: {
    flex: 1,
    ...typography.mono,
    color: colors.textPrimary,
    paddingVertical: spacing.sm + 2,
  },
  dangerText: {
    ...typography.caption,
    color: colors.textMuted,
    lineHeight: 18,
  },
});
